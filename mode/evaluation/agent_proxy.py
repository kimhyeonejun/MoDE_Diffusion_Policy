import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger

import hydra
from lightning_fabric import seed_everything
import numpy as np
from optree import tree_map
import torch

from mode.evaluation.utils import get_default_mode_and_env

from mode.datasets.utils.episode_utils import (
    process_depth,
    process_rgb,
    process_state,
)

logger = getLogger(__name__)

class CalvinAgentWrapper:
    def __init__(
        self,
        agent,
        observation_space_keys,
        proprio_state,
        transforms,
    ):
        self.agent = agent
        self.observation_space_keys = observation_space_keys
        self.proprio_state = proprio_state
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        self.transforms = transforms

    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    def step(self, obs, lang_annotation):
        obs = self._transform_observation(obs)
        obs = tree_map(lambda x: x.to(self.agent.device), obs)
        action = self.agent.step(obs, {'lang_text': lang_annotation})
        action = self._transform_action(action)
        return action

    def _transform_action(self, action_tensor):
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7  # noqa: PLR2004 # these are verbatim copied from calvin
        else:
            if action_tensor.shape[-1] == 7:  # noqa: PLR2004
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:  # noqa: PLR2004
                slice_ids = [3, 7]
            else:
                logger.error(
                    "actions are required to have length 8 (for euler angles) or 9 (for quaternions)"
                )
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)

        action[-1] = 1 if action[-1] > 0 else -1

        return action

    def _transform_observation(self, obs):
        state_obs = process_state(
            obs, self.observation_space_keys, self.transforms, self.proprio_state
        )
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, self.transforms)
        depth_obs = process_depth(
            obs["depth_obs"], self.observation_space_keys, self.transforms
        )

        state_obs["robot_obs"] = state_obs["robot_obs"].unsqueeze(0)
        rgb_obs.update(
            {"rgb_obs": {k: v.unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}}
        )
        depth_obs.update(
            {"depth_obs": {k: v.unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}}
        )

        obs_dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            "robot_obs_raw": torch.from_numpy(obs["robot_obs"]),
        }
        return obs_dict


class AgentHandler(BaseHTTPRequestHandler):
    create_agent = None
    agent = None

    def do_POST(self):  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request = pickle.loads(post_data)  # noqa: S301

        method = request.get("method")

        try:
            if method == "__shutdown__":
                AgentHandler._destroy_agent()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps({"result": "shutdown"}))
                raise KeyboardInterrupt  # To stop the server
            elif method == "__init__":
                AgentHandler.agent = AgentHandler.create_agent()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps({"result": "agent created"}))
            else:
                # pass these to the agent
                agent = AgentHandler.agent
                args = request.get("args", [])
                kwargs = request.get("kwargs", {})
                if method == "__call__":
                    result = agent(*args, **kwargs)
                else:
                    result = getattr(agent, method)(*args, **kwargs)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps({"result": result}))
        except Exception:
            logger.exception("Error handling request")
            AgentHandler._destroy_agent()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(pickle.dumps({"error": "there was a problem"}))

    @staticmethod
    def _destroy_agent():
        del AgentHandler.agent
        AgentHandler.agent = None
        AgentHandler.clear_cuda_cache()
        logger.info("agent destroyed")
        
    @staticmethod
    def clear_cuda_cache():
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()
            # Log memory stats
            for i in range(torch.cuda.device_count()):
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
                reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
                logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")



def start_server(create_agent, host="localhost", port=6000):
    AgentHandler.create_agent = create_agent
    AgentHandler.agent = None
    server = HTTPServer((host, port), AgentHandler)
    logger.info(f"starting server at http://{host}:{port}")  # noqa: G004
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down server")
        server.server_close()

@hydra.main(config_path=str("../../conf"), config_name="agent_proxy")
def main(cfg):
    def create_agent():
        logger.info("loading agent")
        seed_everything(0, workers=True)
        # agent = hydra.utils.instantiate(cfg.model)
        
        # agent = agent.to(cfg.device)
        # agent.eval()

        model, env, data_module, lang_embeddings = get_default_mode_and_env(
            cfg.train_folder,
            cfg.dataset_path,
            cfg.checkpoint,
            env=42,
            lang_embeddings=None,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=cfg.device,
        )
        model = model.to(cfg.device)
        model.eval()

        dataloader = data_module.val_dataloader()
        dataset = dataloader["lang"].dataset

        agent = CalvinAgentWrapper(
            model,
            dataset.observation_space,
            dataset.proprio_state,
            dataset.transforms
        )

        return agent

    start_server(create_agent, cfg.host, cfg.port)


if __name__ == "__main__":
    main()

from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

import cv2
from omegaconf import DictConfig, OmegaConf

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm
import wandb
import torch
import torch.distributed as dist

from mode.evaluation.utils import get_default_mode_and_env
from mode.rollout.rollout_video import RolloutVideo
# Import LIBERO environment utilities
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

logger = logging.getLogger(__name__)


from collections import Counter
from itertools import chain
import logging
import multiprocessing
import os
from typing import Any
import gc


import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from termcolor import colored
import torch
import torch.distributed as dist
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs, evaluate_multitask_training_success
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

# import cv2
# from pathlib import Path
# import sys
# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from mode.evaluation.multistep_sequences import get_sequences
from mode.evaluation.utils import get_env_state_for_initial_condition, join_vis_lang, LangEmbeddings
from mode.rollout.rollout_video import RolloutVideo
from typing import Any, Dict, Tuple, Union


log_print = logging.getLogger(__name__)

def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


class EvaluateLibero:
    def __init__(
        self,
        model,
        transforms,
        log_dir,
        benchmark_name,
        num_sequences,
        max_steps,
        num_videos,
        n_eval,
        task_embedding_format,
        device,
    ):
        self.model = model
        self.transforms = transforms
        self.log_dir = log_dir

        self.device = device
        self.task_order = 0
        self.bddl_folder = get_libero_path("bddl_files")
        self.init_states_folder = get_libero_path("init_states")
        self.task_embedding_format =task_embedding_format
        self.benchmark_name = benchmark_name
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = self.benchmark_dict[self.benchmark_name]()
        self.num_tasks = self.benchmark_instance.get_num_tasks()
        self.num_videos = num_videos
        self.task_names = self.benchmark_instance.get_task_names()
        self.benchmark = get_benchmark(self.benchmark_name)(self.task_order)
        self.n_eval = n_eval
        self.img_h = 224
        self.img_w = 224
        self.rank = None
        self.world_size = None
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        # self.save_dir = save_dir
        self.device = None
        self.eval_sequences = None
        self.init_states_paths = []
        self.cfg = {}
        self.descriptions = []
        self.create_cfg_for_libero(self.task_embedding_format)
        for i in range(self.num_tasks):

            task_i = self.benchmark_instance.get_task(0)

            self.init_states_paths.append(
                os.path.join(self.init_states_folder, self.task_names[i], task_i.init_states_file)
            )
            self.descriptions.append(self.benchmark_instance.get_task(i).language)
            task_embs = get_task_embs(self.cfg, self.descriptions)
            self.benchmark_instance.set_task_embs(task_embs)

        self.all_tasks = list(range(self.benchmark_instance.n_tasks))

    def setup(self) -> None:
        if self.benchmark is None:
            self.eval_sequences = get_sequences(self.num_sequences)
            self.benchmark = get_benchmark(self.benchmark_name)(self.eval_sequences)

    def start(self) -> None:

        successes = self.evaluate_policy(self.model, store_video=self.num_videos)

        result_array = sum(successes) / len(successes)

        # print(f"number of rollouts: {len(successes)}")
        log_print(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
        wandb.log("eval_lh/avg_seq_len", torch.tensor(result_array), on_epoch=True, sync_dist=True)

        for success, task_name in zip(successes, self.task_names):
            log_print(f"eval_lh/sr_{task_name} with success {success}")
            wandb.log(f"eval_lh/sr_{task_name}", success, on_step=False, sync_dist=True)
        print('done')
        print()

    def evaluate_policy(self, model, store_video=False):
        successes = []

        for idx in self.all_tasks:  # Distribute tasks across GPUs
            task_name = self.task_names[idx]
            task_i = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            log_print(f"starting to evaluate: {task_name}")
            success_rate = self.evaluate_task(model, task_i, task_emb, task_str, idx, store_video=store_video)
            successes.append(success_rate)

        return successes

    def evaluate_task(self, model, task_i, task_emb, task_str, idx, sim_states=None, store_video=0):
        env_args = {
            "bddl_file_name": os.path.join(
                self.bddl_folder, task_i.problem_folder, task_i.bddl_file
            ),
            "camera_heights": self.img_h,
            "camera_widths": self.img_w,
        }

        # Try to handle the frame buffer issue
        env_creation = False
        count = 0
        while not env_creation and count < 5:
            try:
                env = OffScreenRenderEnv(**env_args)
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            self.init_states_folder, task_i.problem_folder, task_i.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in tqdm(range(self.n_eval), desc="Evaluating"):
            store_video_this_rollout = i < store_video
            if store_video_this_rollout:
                video_frames = []
                video_filename = f"rollout_{task_str}_nmp_{i}.mp4"
                video_path = os.path.join(self.log_dir, video_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (self.img_w, self.img_h))

            env.reset()

            done = False
            steps = 0
            model.reset()
            obs = env.set_init_state(init_states)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros(7)
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                sim_state = env.get_sim_state()
                if sim_states is not None:
                    sim_states[i].append(sim_state)

            while steps < self.max_steps:
                steps += 1
                data, goal = self.process_env_obs(obs, task_emb, task_i.language)
                # data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = model.step(data, goal)
                actions = actions.cpu().numpy()
                obs, reward, done, info = env.step(actions)

                if store_video_this_rollout:
                    video_frames.append(obs['agentview_image'])

                if done:
                    break

            if store_video_this_rollout:
                for frame in video_frames:
                    video_writer.write(frame)
                video_writer.release()

            # a new form of success record
            num_success += int(done)


        success_rate = num_success / self.n_eval
        env.close()
        gc.collect()
        # print(f"[info] evaluate task {task_str} takes {t.get_elapsed_time():.1f} seconds")
        return success_rate

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({'task_embedding_format': task_embedding_format,
                               'data': {'max_word_len': 25}})

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()


    def translate_obs_space(self, obs_space):

        translated_dict = {}
        translated_dict['rgb_obs'] = {}
        translated_dict['rgb_obs']['rgb_static'] = obs_space['agentview_image']
        translated_dict["rgb_obs"]['rgb_gripper'] = obs_space['robot0_eye_in_hand_image']
        translated_dict['robot_obs'] = obs_space['robot0_joint_pos']
        translated_dict['gripper_states'] = obs_space['robot0_gripper_qpos']
        translated_dict['depth_obs'] = {}

        return translated_dict

    def apply_transforms(self, data, train=False):
        # Assuming data contains images in 'rgb_static' and 'rgb_gripper'
        for key in data['rgb_obs']:
            # print(key)
            x = data['rgb_obs'][key]
            if len(x.shape) == 3:
                x = np.expand_dims(x, axis=0)
                # print(x.shape)
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            for transform in self.transforms[key]:
                x = transform(x)
            data['rgb_obs'][key] = x.unsqueeze(0).to(self.device)
            # data['rgb_obs'][key] = transforms[key](data['rgb_obs'][key])

        return data

    def process_env_obs(self, env_obs, lang_embed, lang_text=None):
        return_obs = self.translate_obs_space(env_obs)
        return_obs = self.apply_transforms(return_obs)

        goal = {}
        goal['lang_text'] = lang_text
        goal['lang'] = lang_embed

        return return_obs, goal

@hydra.main(config_path="../../conf", config_name="mode_evaluate_libero")
def main(cfg):
    seed_everything(0, workers=True)
    model, _, dm, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=42,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
        prep_dm_and_deps=False
    )

    model = model.to(cfg.device)
    model.eval()

    log_dir = get_log_dir(cfg.log_dir)
    # transforms = dm.val_dataloader().dataset.transforms
    transforms = hydra.utils.instantiate(dm.transforms)

    eval_libero = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        benchmark_name=cfg.benchmark_name,
        num_sequences=cfg.num_sequences,
        num_videos=cfg.num_videos,
        max_steps=cfg.max_steps,
        n_eval=cfg.n_eval,
        task_embedding_format=cfg.task_embedding_format,
        device=cfg.device,
    )

    if cfg.log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=False)
        run = wandb.init(
            project='mode_libero_eval',
            entity=cfg.wandb_entity,
            config=OmegaConf.to_object(cfg),
        )

    if cfg.log_wandb:
        run.finish()


if __name__ == "__main__":
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    main()
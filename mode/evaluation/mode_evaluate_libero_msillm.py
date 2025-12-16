from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
import gc
from typing import Any, Dict, Tuple, Union

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
import torch
import wandb

# This is for using the locally installed repo clone when using slurm
repo_root = Path(__file__).absolute().parents[2]
sys.path.insert(0, repo_root.as_posix())

# Add LIBERO submodule to path so 'libero' module can be imported
libero_repo_dir = repo_root / "LIBERO"
if libero_repo_dir.exists():
    sys.path.insert(0, str(libero_repo_dir))
    # Also set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{libero_repo_dir}:{current_pythonpath}" if current_pythonpath else str(libero_repo_dir)

from mode.evaluation.utils import get_msillm_mode_and_env
from mode.evaluation.multistep_sequences import get_sequences
from mode.rollout.rollout_video import RolloutVideo
from mode.utils.bpp_utils import (
    calculate_bpp_from_encoder_output,
    accumulate_bpp_stats,
    compute_average_bpp,
)
# Import LIBERO environment utilities
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)


log_print = logging.getLogger(__name__)

def get_log_dir(log_dir, checkpoint_name=None):
    # Use checkpoint-based directory if checkpoint name is provided
    if checkpoint_name:
        checkpoint_stem = Path(checkpoint_name).stem  # Remove .ckpt extension
        log_dir = Path("outputs") / checkpoint_stem
        os.makedirs(log_dir, exist_ok=True)
    else:
        # Fallback to Hydra's output directory or specified log_dir
        hydra_output_dir = Path.cwd()
        if (hydra_output_dir / ".hydra").exists():
            # We're running under Hydra, use its output directory
            log_dir = hydra_output_dir
        elif log_dir is not None:
            log_dir = Path(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = Path(__file__).parents[3] / "outputs" / "libero_eval"
            os.makedirs(log_dir, exist_ok=True)
    
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

        # Normalize device to torch.device
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.task_order = 0
        self.bddl_folder = get_libero_path("bddl_files")
        self.init_states_folder = get_libero_path("init_states")
        self.task_embedding_format = task_embedding_format
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
        self.eval_sequences = None
        self.init_states_paths = []
        self.cfg = {}
        self.descriptions = []
        
        # BPP statistics tracking
        self.bpp_stats = {}
        
        # First, collect all descriptions
        for i in range(self.num_tasks):
            task_i = self.benchmark_instance.get_task(0)
            self.init_states_paths.append(
                os.path.join(self.init_states_folder, self.task_names[i], task_i.init_states_file)
            )
            self.descriptions.append(self.benchmark_instance.get_task(i).language)

        # Now create cfg and task embeddings with descriptions available
        self.create_cfg_for_libero(self.task_embedding_format)
        
        # Use task embeddings created in create_cfg_for_libero
        if hasattr(self, 'task_embs'):
            self.benchmark_instance.set_task_embs(self.task_embs)
        else:
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

        # Print results to console
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Average success rate: {result_array:.4f} ({result_array*100:.2f}%)")
        print(f"Number of tasks: {len(successes)}")
        print(f"\nPer-task success rates:")
        for success, task_name in zip(successes, self.task_names):
            print(f"  {task_name}: {success:.4f} ({success*100:.2f}%)")
        print(f"{'='*60}\n")

        # Save results to JSON file
        results_dict = {
            "average_success_rate": float(result_array),
            "num_tasks": len(successes),
            "per_task_success": {
                task_name: float(success) for success, task_name in zip(successes, self.task_names)
            }
        }
        results_file = self.log_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {results_file}")

        # Also log to logger
        log_print.info(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
        if wandb.run is not None:
            wandb.log({"eval_lh/avg_seq_len": torch.tensor(result_array)})

        for success, task_name in zip(successes, self.task_names):
            log_print.info(f"eval_lh/sr_{task_name} with success {success}")
            if wandb.run is not None:
                wandb.log({f"eval_lh/sr_{task_name}": success})
        
        # Log BPP statistics if available
        if self.bpp_stats:
            avg_bpp = compute_average_bpp(self.bpp_stats)
            print(f"\n{'='*60}")
            print(f"BPP Statistics:")
            print(f"{'='*60}")
            for key, bpp_value in avg_bpp.items():
                print(f"  {key}: {bpp_value:.4f} bpp")
                log_print.info(f"bpp/{key}: {bpp_value:.4f}")
                if wandb.run is not None:
                    wandb.log({f"bpp/{key}": bpp_value})
            print(f"{'='*60}\n")
            
            # Add to results dict
            results_dict["bpp"] = {k: float(v) for k, v in avg_bpp.items()}
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)

    def evaluate_policy(self, model, store_video=False):
        successes = []

        for idx in self.all_tasks:  # Distribute tasks across GPUs
            task_name = self.task_names[idx]
            task_i = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            log_print.info(f"starting to evaluate: {task_name}")
            success_rate = self.evaluate_task(model, task_i, task_emb, task_str, idx, store_video=store_video)
            successes.append(success_rate)

        return successes

    def evaluate_task(self, model, task_i, task_emb, task_str, idx, sim_states=None, store_video=0):
        # Setup BPP measurement by wrapping encoder to capture latents
        msillm_model = getattr(model, "msillm_model", None)
        encoder = None
        original_encoder_forward = None
        
        if msillm_model is not None:
            encoder = getattr(msillm_model, "encoder", None)
            if encoder is not None:
                # Store original forward method
                original_encoder_forward = encoder.forward
                
                # Wrap encoder to capture latents when called
                def _capture_encoder_forward(*args, **kwargs):
                    latent = original_encoder_forward(*args, **kwargs)
                    # Store latent for BPP calculation
                    # We'll identify which image it is by checking the input shape or using a counter
                    if not hasattr(_capture_encoder_forward, 'call_count'):
                        _capture_encoder_forward.call_count = 0
                        _capture_encoder_forward.latents = []
                    
                    _capture_encoder_forward.call_count += 1
                    _capture_encoder_forward.latents.append(latent)
                    return latent
                
                encoder.forward = _capture_encoder_forward
        
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
        # Inference-only evaluation
        with torch.inference_mode():
            for i in tqdm(range(self.n_eval), desc="Evaluating"):
                try:
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

                    # Use per-rollout init state if a list is provided
                    state = init_states
                    if isinstance(init_states, (list, tuple)):
                        state = init_states[i % len(init_states)]

                    # Some saved init states are numpy arrays that do not match MuJoCo's expected (MjData, float) signature.
                    # To avoid type errors during evaluation, always fall back to a clean reset.
                    try:
                        obs = env.reset()
                    except Exception as e:
                        log_print.warning(f"env.reset() failed on rollout {i}: {e}; retrying after short sleep")
                        time.sleep(1)
                        obs = env.reset()

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
                        
                        # Clear captured latents for this step
                        if encoder is not None and hasattr(encoder.forward, 'latents'):
                            encoder.forward.latents = []
                            encoder.forward.call_count = 0
                        
                        actions = model.step(data, goal)
                        
                        # Calculate BPP from captured latents (compression happened in model.step)
                        if encoder is not None and hasattr(encoder.forward, 'latents') and len(encoder.forward.latents) > 0:
                            bpp_dict = {}
                            # encoder.forward.latents contains latents in order: [rgb_static_latent, rgb_gripper_latent]
                            # Each latent corresponds to one image after reshape to (b*t, c, h, w)
                            keys = ['rgb_static', 'rgb_gripper']
                            for idx_latent, key in enumerate(keys):
                                if idx_latent < len(encoder.forward.latents):
                                    latent = encoder.forward.latents[idx_latent]
                                    # Get original image shape from data
                                    if key in data.get('rgb_obs', {}):
                                        img = data['rgb_obs'][key]  # (1, C, H, W)
                                        # Remove batch dim for BPP calculation
                                        img_for_bpp = img.squeeze(0)  # (C, H, W)
                                        
                                        # Calculate BPP
                                        bpp = calculate_bpp_from_encoder_output(
                                            latent,
                                            img_for_bpp,
                                            bits_per_element=8,  # Assuming 8-bit quantization
                                        )
                                        bpp_dict[key] = bpp
                            
                            # Accumulate BPP statistics
                            if bpp_dict:
                                self.bpp_stats = accumulate_bpp_stats(bpp_dict, self.bpp_stats)
                        
                        actions = actions.cpu().numpy()
                        obs, reward, done, info = env.step(actions)

                        if store_video_this_rollout:
                            frame = obs['agentview_image']
                            # Fix: Rotate 180 degrees and convert BGR to RGB
                            # Rotate 180 degrees on height and width axes
                            frame = np.rot90(frame, k=2, axes=(0, 1))  # Rotate on h, w axes
                            # Convert BGR to RGB by reversing the channel dimension
                            frame = frame[..., ::-1]  # Reverse channels: BGR -> RGB
                            video_frames.append(frame)

                        if done:
                            break

                    if store_video_this_rollout:
                        for frame in video_frames:
                            video_writer.write(frame)
                        video_writer.release()

                    # a new form of success record
                    num_success += int(done)
                except Exception as e:
                    import traceback
                    error_msg = f"Error during rollout {i} of task {task_str}: {e}"
                    error_traceback = traceback.format_exc()
                    # Use print and tqdm.write to ensure error messages are visible
                    print(f"\n{'='*60}")
                    print(f"ERROR: {error_msg}")
                    print(f"{'='*60}")
                    print(error_traceback)
                    print(f"{'='*60}\n")
                    # Also log to logger
                    log_print.error(error_msg)
                    log_print.error(error_traceback)
                    # Continue to next rollout even if this one failed
                    continue

        success_rate = num_success / self.n_eval
        
        # Restore original encoder forward if it was wrapped
        if encoder is not None and original_encoder_forward is not None:
            encoder.forward = original_encoder_forward
        
        env.close()
        gc.collect()
        return success_rate

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({
            'task_embedding_format': task_embedding_format,
            'data': {'max_word_len': 25},
            'task_embedding_one_hot_offset': 1
        })

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()

        # Create task embeddings - same as training: use get_task_embs
        # If that fails, fall back to 2-dim random embeddings
        import torch
        num_tasks = len(self.descriptions)
        
        try:
            # Use get_task_embs same as training (libero_dataset.py line 147)
            task_embs = get_task_embs(self.cfg, self.descriptions)
            self.task_embs = task_embs
            print(f"Created {num_tasks} {task_embedding_format} task embeddings using get_task_embs (dim={task_embs[0].shape[0]})")
            return
        except Exception as e:
            print(f"Warning: Failed to load task embeddings using get_task_embs: {e}")
            print("Falling back to 2-dim random embeddings")
            task_embs = []
            for i in range(num_tasks):
                emb = torch.randn(2, device=self.device)
                task_embs.append(emb)
            self.task_embs = task_embs
            print(f"Created {num_tasks} random task embeddings (dim=2)")
            return


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
        # Ensure lang_embed is on the correct device
        # if isinstance(lang_embed, torch.Tensor):
        #     goal['lang'] = lang_embed.to(self.device)
        # else:
        #     # If it's numpy or list, convert to tensor first
        #     if isinstance(lang_embed, np.ndarray):
        #         goal['lang'] = torch.from_numpy(lang_embed).to(self.device).float()
        #     else:
        #         goal['lang'] = torch.tensor(lang_embed, device=self.device).float()

        return return_obs, goal

@hydra.main(config_path="../../conf", config_name="mode_evaluate_libero_msillm")
def main(cfg: DictConfig):
    seed_everything(0, workers=True)
    
    # Handle checkpoint path from environment variable if provided
    checkpoint_env = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_env:
        print(f"Using checkpoint from environment variable: {checkpoint_env}")
        cfg.checkpoint = checkpoint_env
    
    # Sanitize checkpoint filename: replace '=' with '-' to avoid Hydra parsing issues
    if "=" in cfg.checkpoint and not Path(cfg.checkpoint).is_absolute():
        # Only sanitize if it's a relative path (filename only)
        sanitized_checkpoint = cfg.checkpoint.replace("=", "-")
        checkpoint_path = Path(cfg.train_folder) / cfg.checkpoint
        sanitized_path = Path(cfg.train_folder) / sanitized_checkpoint
        
        # If sanitized file exists, use it; otherwise try original
        if sanitized_path.exists():
            print(f"Using sanitized checkpoint path: {sanitized_checkpoint}")
            cfg.checkpoint = sanitized_checkpoint
        elif checkpoint_path.exists():
            print(f"Warning: Checkpoint filename contains '=' which may cause Hydra parsing issues.")
            print(f"Consider renaming: {checkpoint_path} -> {sanitized_path}")
            print(f"Using original checkpoint path: {cfg.checkpoint}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    # Hydra creates outputs/${hydra.job.name}/ directory
    # We need to rename/move it to use checkpoint name without extension
    # But Hydra already created the directory, so we'll work with what we have
    # and ensure log_dir uses the checkpoint-based name
    
    # Handle CUDA_VISIBLE_DEVICES properly for torch device selection
    # When CUDA_VISIBLE_DEVICES is set, physical GPU N becomes logical GPU 0 (for CUDA).
    # So we always use cuda:0 when CUDA_VISIBLE_DEVICES is set.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        device_id = 0  # Logical GPU 0 (maps to physical GPU specified in CUDA_VISIBLE_DEVICES)
        device_str = "cuda:0"
    else:
        # Use cfg.device if available, otherwise default to cuda:0
        device_id = getattr(cfg, 'device', 0)
        if isinstance(device_id, str):
            device_str = device_id
        else:
            device_str = f"cuda:{device_id}"
    
    print(f"Using device: {device_str} (CUDA_VISIBLE_DEVICES={cuda_visible})")
    
    # Load model using utility function (handles MS-ILLM automatically)
    model, _, dm, _, loaded_cfg = get_msillm_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite if hasattr(cfg, 'eval_cfg_overwrite') else {},
        device_id=device_id,
        prep_dm_and_deps=False
    )
    
    model.eval()

    # Get log directory based on checkpoint name (without extension)
    log_dir = get_log_dir(cfg.log_dir, checkpoint_name=cfg.checkpoint)
    
    # Get transforms from loaded config (use validation transforms if available)
    try:
        transforms_cfg = loaded_cfg.datamodule.transforms.val if "val" in loaded_cfg.datamodule.transforms else loaded_cfg.datamodule.transforms
        transforms = hydra.utils.instantiate(transforms_cfg)
    except (AttributeError, KeyError):
        # Fallback: try from datamodule if available
        if hasattr(dm, 'transforms') and dm.transforms is not None:
            transforms = hydra.utils.instantiate(dm.transforms)
        else:
            # Final fallback: use main config
            transforms_cfg = cfg.datamodule.transforms.val if "val" in cfg.datamodule.transforms else cfg.datamodule.transforms
            transforms = hydra.utils.instantiate(transforms_cfg)

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
        device=device_str,
    )

    # Setup wandb logger
    if cfg.log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=True)
        checkpoint_stem = Path(cfg.checkpoint).stem
        # Resolve interpolations before converting to dict for wandb config
        wandb_config = {
            "checkpoint": cfg.checkpoint,
            "benchmark_name": cfg.benchmark_name,
            "num_sequences": cfg.num_sequences,
            "n_eval": cfg.n_eval,
            "max_steps": cfg.max_steps,
        }

        # Backward/forward compatible wandb config:
        # - Some configs provide `logger.*`
        # - Older configs provide `wandb_entity` only (no `logger` key, often with struct mode enabled)
        project = OmegaConf.select(cfg, "logger.project", default=None) or "mode_libero_eval"
        group = OmegaConf.select(cfg, "logger.group", default=None) or "mode_libero_eval"
        mode = OmegaConf.select(cfg, "logger.mode", default=None) or "online"
        run_id = OmegaConf.select(cfg, "logger.id", default=None)
        entity = OmegaConf.select(cfg, "logger.entity", default=None)
        if entity in ("null", "", None):
            # legacy key
            entity = cfg.get("wandb_entity", None)
            if entity in ("null", ""):
                entity = None
        if run_id in ("null", ""):
            run_id = None

        run = wandb.init(
            project=project,
            entity=entity,
            name=checkpoint_stem,
            group=group,
            config=wandb_config,
            dir=str(log_dir / "wandb"),
            mode=mode,
            id=run_id,
            allow_val_change=True,
        )

    # Actually run the evaluation!
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    eval_libero.setup()
    eval_libero.start()

    if cfg.log_wandb and wandb.run is not None:
        wandb.finish()
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == "__main__":
    main()
import json
import logging
import os
from pathlib import Path
import sys
import time
import gc

import hydra
import imageio
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
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

from mode.evaluation.utils import get_msillm_mode_and_env, reconstruct_frame_for_video
from mode.evaluation.multistep_sequences import get_sequences
from mode.utils.bpp_utils import (
    calculate_bpp_from_hyperprior_output,
    accumulate_bpp_stats,
    compute_average_bpp,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.utils import get_task_embs


log_print = logging.getLogger(__name__)


def _get_bpp_wrapper(model):
    """Helper to get BPP wrapper from model."""
    wrapper = getattr(model, "_bpp_wrapper", None)
    if wrapper is None:
        msillm_model = getattr(model, "msillm_model", None)
        if msillm_model is not None:
            if hasattr(msillm_model, "compress") and isinstance(msillm_model.compress, LatentCaptureWrapper):
                wrapper = msillm_model.compress
    return wrapper


def _calculate_bpp_from_latents(bpp_wrapper, data, sensors):
    """Calculate BPP from captured latents."""
    if bpp_wrapper is None or len(bpp_wrapper.latents) == 0:
        return {}
    bpp_dict = {}
    latents = bpp_wrapper.latents[-len(sensors):]
    for sensor_name, latent in zip(sensors, latents):
        img = data["rgb_obs"][sensor_name].squeeze(0)  # (C, H, W)
        bpp = calculate_bpp_from_hyperprior_output(latent, img.shape)
        bpp_dict[sensor_name] = bpp
    return bpp_dict


def _prepare_video_frame(model, obs, store_reconstructed, sensor_name='rgb_static', data=None):
    """Prepare video frame from model or observation.
    
    Args:
        model: Model with reconstructed frame tensors
        obs: Environment observation dictionary
        store_reconstructed: Whether to use reconstructed frames
        sensor_name: 'rgb_static' or 'rgb_gripper'
        data: Optional transformed data dictionary (to use transforms-processed images)
    """
    if store_reconstructed:
        # Check for sensor-specific reconstructed frame
        tensor_attr = f'_last_reconstructed_frame_tensor_{sensor_name}'
        if hasattr(model, tensor_attr):
            recon_frame = getattr(model, tensor_attr)
            if recon_frame is not None:
                rgb_recon_np = (recon_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb_recon_np = np.rot90(rgb_recon_np, k=2, axes=(0, 1))
                return rgb_recon_np[..., ::-1]  # RGB to BGR
        # If reconstructed frame is not available (e.g., compress_rgb=False), fall through to use data/obs
    
    # Use transforms-processed image if available (already in [0, 1] range, no denormalize needed)
    if data is not None and sensor_name in data.get("rgb_obs", {}):
        img_tensor = data["rgb_obs"][sensor_name].squeeze(0)[0]  # [1, T, C, H, W] -> [C, H, W]
        frame = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame = np.rot90(frame, k=2, axes=(0, 1))
        return frame[..., ::-1]  # RGB to BGR
    
    # # Fallback: Use original observation
    # frame = obs['agentview_image'] if sensor_name == 'rgb_static' else obs['robot0_eye_in_hand_image']
    # frame = np.rot90(frame, k=2, axes=(0, 1)) if isinstance(frame, np.ndarray) else frame
    # return frame[..., ::-1]


class LatentCaptureWrapper:
    """Simple wrapper to capture latents for BPP calculation."""
    def __init__(self, original_method):
        self.original = original_method
        self.latents = []
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        latent = self.original(*args, **kwargs)
        self.latents.append(latent)
        self.call_count += 1
        return latent
    
    def clear(self):
        """Clear captured latents."""
        self.latents = []
        self.call_count = 0

def get_log_dir(log_dir, checkpoint_name=None):
    """
    Resolve evaluation output directory.
    
    Behavior:
    - If `checkpoint_name` is provided: use it as a subdirectory name under a *base* directory.
      The base directory is `log_dir` (if provided), otherwise Hydra's run dir (if active),
      otherwise a sensible default under the repo.
    - If `checkpoint_name` is not provided: use Hydra's run dir (if active), else `log_dir`,
      else default under the repo.
    """
    hydra_output_dir = Path.cwd()
    running_under_hydra = (hydra_output_dir / ".hydra").exists()

    # Choose base directory
    if log_dir is not None:
        base_dir = Path(log_dir)
    elif running_under_hydra:
        base_dir = hydra_output_dir
    else:
        base_dir = Path(__file__).parents[3] / "outputs" / "libero_eval"

    # If checkpoint_name is provided, namespace outputs under base_dir/<checkpoint_path_without_suffix>
    # e.g. checkpoint_name="zero_padding_sam/foo/bar/epoch=02.ckpt"
    #   -> base_dir/"zero_padding_sam/foo/bar/epoch=02"
    if checkpoint_name:
        ckpt_path = Path(checkpoint_name)
        # Avoid mirroring absolute paths into the output directory tree.
        if ckpt_path.is_absolute():
            ckpt_subdir = Path(ckpt_path.stem)
        else:
            ckpt_subdir = ckpt_path.with_suffix("")  # keeps subdirectories, drops extension
        log_dir = base_dir / ckpt_subdir
    else:
        log_dir = base_dir

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
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.eval_sequences = None
        self.cfg = {}
        self.descriptions = []
        
        # BPP statistics tracking
        self.bpp_stats = {}
        
        # First, collect all descriptions
        for i in range(self.num_tasks):
            self.descriptions.append(self.benchmark_instance.get_task(i).language)

        # Now create cfg and task embeddings with descriptions available
        self.create_cfg_for_libero(self.task_embedding_format)
        
        # Set task embeddings
        task_embs = getattr(self, 'task_embs', None)
        if task_embs is None:
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
        if self.bpp_stats and len(self.bpp_stats) > 0:
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
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation of {len(self.all_tasks)} tasks")
        print(f"{'='*60}\n")

        for idx in self.all_tasks:  # Distribute tasks across GPUs
            task_name = self.task_names[idx]
            task_i = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            log_print.info(f"starting to evaluate: {task_name}")
            print(f"\n[{idx+1}/{len(self.all_tasks)}] Evaluating: {task_name}")
            print(f"Task description: {task_i.language}")
            success_rate = self.evaluate_task(model, task_i, task_emb, task_str, idx, store_video=store_video)
            successes.append(success_rate)
            
            # Print immediate result for this task
            print(f"\n✓ Task {idx+1}/{len(self.all_tasks)} completed: {task_name}")
            print(f"  Success rate: {success_rate:.2%} ({success_rate*self.n_eval:.0f}/{self.n_eval})")
            
            # Print running average
            if len(successes) > 0:
                avg_success = sum(successes) / len(successes)
                print(f"  Running average: {avg_success:.2%} across {len(successes)} tasks")
            print()

        return successes

    def evaluate_task(self, model, task_i, task_emb, task_str, idx, sim_states=None, store_video=0):
        bpp_wrapper = _get_bpp_wrapper(model)
        if bpp_wrapper is None:
            log_print.warning(f"[BPP] No BPP wrapper found for task {task_str} - BPP will not be measured")
        else:
            log_print.info(f"[BPP] Found BPP wrapper for task {task_str}")
        
        env_args = {
            "bddl_file_name": os.path.join(
                self.bddl_folder, task_i.problem_folder, task_i.bddl_file
            ),
            "camera_heights": self.img_h,
            "camera_widths": self.img_w,
        }
        compress_gripper = getattr(model, '_compress_gripper', True)
        compress_rgb = getattr(model, '_compress_rgb', True)

        # Try to handle the frame buffer issue
        env_creation = False
        count = 0
        last_error = None
        while not env_creation and count < 5:
            try:
                env = OffScreenRenderEnv(**env_args)
                env_creation = True
            except Exception as e:
                last_error = e
                time.sleep(5)
                count += 1
        if count >= 5:
            raise RuntimeError(f"Failed to create environment after 5 attempts: {last_error}") from last_error

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            self.init_states_folder, task_i.problem_folder, task_i.init_states_file
        )
        init_states = torch.load(init_states_path, weights_only=False)
        num_success = 0
        pbar = tqdm(range(self.n_eval), desc=f"Evaluating {task_i.language[:30]}")
        for i in pbar:
            store_video_this_rollout = i < store_video
            if store_video_this_rollout:
                video_frames_static = []
                video_frames_gripper = []
                video_filename_static = f"rollout_{task_str}_static_nmp_{i}.mp4"
                video_filename_gripper = f"rollout_{task_str}_gripper_nmp_{i}.mp4"
                video_path_static = os.path.join(self.log_dir, video_filename_static)
                video_path_gripper = os.path.join(self.log_dir, video_filename_gripper)
                fps = 20.0  # Frame rate

            env.reset()

            done = False
            steps = 0
            model.reset()
            # Select one init state for this rollout (same as LIBERO's metric.py)
            init_state_idx = i % init_states.shape[0]
            init_state = init_states[init_state_idx]
            obs = env.set_init_state(init_state)

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
                if bpp_wrapper is not None:
                    bpp_wrapper.clear()
                
                actions = model.step(data, goal)
                
                # Fix for stuttering video: if model.step() didn't run inference (action chunking),
                # manually reconstruct frame for smooth video.
                if (store_video_this_rollout 
                    and hasattr(model, '_store_reconstructed_frame') 
                    and model._store_reconstructed_frame):
                    # Reconstruct rgb_static if available and compress_rgb is enabled
                    if "rgb_static" in data["rgb_obs"] and compress_rgb:
                        recon_frame = reconstruct_frame_for_video(model, data["rgb_obs"]["rgb_static"])
                        if recon_frame is not None:
                            model._last_reconstructed_frame_tensor_rgb_static = recon_frame
                    # Reconstruct rgb_gripper if available and compress_gripper is enabled
                    if "rgb_gripper" in data["rgb_obs"] and compress_gripper:
                        recon_frame_gripper = reconstruct_frame_for_video(model, data["rgb_obs"]["rgb_gripper"])
                        if recon_frame_gripper is not None:
                            model._last_reconstructed_frame_tensor_rgb_gripper = recon_frame_gripper

                # Calculate BPP from captured latents
                sensors = []
                if compress_rgb and "rgb_static" in data.get("rgb_obs", {}):
                    sensors.append("rgb_static")
                if compress_gripper and "rgb_gripper" in data.get("rgb_obs", {}):
                    sensors.append("rgb_gripper")
                bpp_dict = _calculate_bpp_from_latents(bpp_wrapper, data, sensors)
                if bpp_dict:
                    self.bpp_stats = accumulate_bpp_stats(bpp_dict, self.bpp_stats)
                
                actions = actions.cpu().numpy()
                obs, reward, done, info = env.step(actions)

                if store_video_this_rollout:
                    store_reconstructed = (hasattr(model, '_store_reconstructed_frame') 
                                         and model._store_reconstructed_frame)
                    # Save static frame (use data for transforms-processed image if no reconstructed frame)
                    frame_static = _prepare_video_frame(model, obs, store_reconstructed, sensor_name='rgb_static', data=data)
                    video_frames_static.append(frame_static)
                    # Save gripper frame (use data for transforms-processed image if no reconstructed frame)
                    frame_gripper = _prepare_video_frame(model, obs, store_reconstructed, sensor_name='rgb_gripper', data=data)
                    video_frames_gripper.append(frame_gripper)

                if done:
                    break

            if store_video_this_rollout:
                # Write static video using imageio
                if len(video_frames_static) > 0:
                    video_writer_static = imageio.get_writer(video_path_static, fps=fps)
                    for frame in video_frames_static:
                        # Convert BGR to RGB for imageio
                        frame_rgb = frame[..., ::-1]
                        video_writer_static.append_data(frame_rgb)
                    video_writer_static.close()
                    print(f"[Video] Saved static video: {video_path_static} ({len(video_frames_static)} frames)")
                else:
                    print(f"[Video] Warning: No static frames captured for video")
                
                # Write gripper video using imageio
                if len(video_frames_gripper) > 0:
                    video_writer_gripper = imageio.get_writer(video_path_gripper, fps=fps)
                    for frame in video_frames_gripper:
                        # Convert BGR to RGB for imageio
                        frame_rgb = frame[..., ::-1]
                        video_writer_gripper.append_data(frame_rgb)
                    video_writer_gripper.close()
                    print(f"[Video] Saved gripper video: {video_path_gripper} ({len(video_frames_gripper)} frames)")
                else:
                    print(f"[Video] Warning: No gripper frames captured for video")

            # a new form of success record
            num_success += int(done)
            
            # Update progress bar with current success rate
            current_success_rate = num_success / (i + 1)
            pbar.set_postfix({
                'success': num_success,
                'total': i + 1,
                'rate': f'{current_success_rate:.1%}',
                'status': '✓' if done else '✗'
            })
            
            # Log each rollout result
            log_print.info(f"Rollout {i+1}/{self.n_eval}: {'SUCCESS' if done else 'FAILED'} (current rate: {current_success_rate:.2%}, {num_success}/{i+1})")

        success_rate = num_success / self.n_eval
        pbar.close()
        
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
        import torch
        num_tasks = len(self.descriptions)
        
        # Use get_task_embs same as training (libero_dataset.py line 147)
        task_embs = get_task_embs(self.cfg, self.descriptions)
        self.task_embs = task_embs
        print(f"Created {num_tasks} {task_embedding_format} task embeddings using get_task_embs (dim={task_embs[0].shape[0]})")
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
        for key in data['rgb_obs']:
            x = data['rgb_obs'][key]
            if len(x.shape) == 3:
                x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            for transform in self.transforms[key]:
                x = transform(x)
            data['rgb_obs'][key] = x.unsqueeze(0).to(self.device)
        return data

    def process_env_obs(self, env_obs, lang_embed, lang_text=None):
        return_obs = self.translate_obs_space(env_obs)
        return_obs = self.apply_transforms(return_obs)

        goal = {}
        # Ensure lang_text is a list (lang_buffer expects list)
        if lang_text is not None:
            if isinstance(lang_text, str):
                goal['lang_text'] = [lang_text]
            else:
                goal['lang_text'] = lang_text
        else:
            goal['lang_text'] = None
        goal['lang'] = lang_embed
        return return_obs, goal

def _instantiate_transforms(transforms_cfg):
    transforms = {}
    for key, t_cfg in transforms_cfg.items():
        if isinstance(t_cfg, (list, ListConfig)):
            t_list = [hydra.utils.instantiate(t) for t in t_cfg]
            transforms[key] = t_list
        else:
            transforms[key] = [hydra.utils.instantiate(t_cfg)]
    return transforms


def _load_transforms(loaded_cfg, dm, cfg):
    """Load transforms with fallback chain."""
    # Try from loaded config first
    if hasattr(loaded_cfg, 'datamodule') and hasattr(loaded_cfg.datamodule, 'transforms'):
        transforms_cfg = loaded_cfg.datamodule.transforms.get("val", loaded_cfg.datamodule.transforms)
    # Fallback to DM transforms
    elif hasattr(dm, 'transforms') and dm.transforms is not None:
        transforms_cfg = dm.transforms.get('val', dm.transforms) if isinstance(dm.transforms, dict) else dm.transforms
    # Final fallback to current cfg
    else:
        transforms_cfg = cfg.datamodule.transforms.get("val", cfg.datamodule.transforms)
    
    return _instantiate_transforms(transforms_cfg)


def _get_device_config(cfg):
    """Get device ID and device string from config."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return 0, "cuda:0"
    
    device_id = getattr(cfg, 'device', 0)
    device_str = device_id if isinstance(device_id, str) else f"cuda:{device_id}"
    return device_id, device_str


def _setup_msillm_features(model, cfg):
    """Setup MS-ILLM BPP measurement wrapper and video settings."""
    # Wrap MS-ILLM compression method for BPP measurement
    msillm_model = getattr(model, "msillm_model", None)
    wrapper = None
    if msillm_model is not None and hasattr(msillm_model, "compress"):
        wrapper = LatentCaptureWrapper(msillm_model.compress)
        msillm_model.compress = wrapper
        print(f"[BPP] Wrapped msillm_model.compress for BPP measurement")
    
    if wrapper is not None:
        model._bpp_wrapper = wrapper
        print(f"[BPP] Stored wrapper on model._bpp_wrapper")
    else:
        print(f"[BPP] WARNING: No wrapper created - BPP measurement may not work")
    
    # Enable storing reconstructed frames for video
    use_reconstructed = getattr(cfg, 'use_reconstructed_video', True)
    has_msillm = hasattr(model, 'msillm_model') and model.msillm_model is not None
    
    if has_msillm and use_reconstructed:
        model._store_reconstructed_frame = True
        print("[Video] Will save MS-ILLM reconstructed images to video")
    else:
        model._store_reconstructed_frame = False
        if has_msillm:
            print("[Video] Will save original env images to video (use_reconstructed_video=False)")
        else:
            print("[Video] Will save original env images to video (no MS-ILLM model)")


def _resolve_checkpoint_path(cfg):
    """Resolve checkpoint path from env var or config."""
    checkpoint_env = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_env:
        print(f"Using checkpoint from environment variable: {checkpoint_env}")
        cfg.checkpoint = checkpoint_env
        return
    
    # If checkpoint is not specified, use pretrain_chk from config
    if not cfg.checkpoint or cfg.checkpoint in ("", "null", None):
            if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.initialize("../../conf")
            base_cfg = hydra.compose(config_name="config_libero_msillm")
            if hasattr(base_cfg, "pretrain_chk") and base_cfg.pretrain_chk:
                cfg.checkpoint = base_cfg.pretrain_chk
                print(f"No checkpoint specified, using pretrained checkpoint: {cfg.checkpoint}")
            else:
                raise ValueError("No checkpoint specified and pretrain_chk not found in config")


def _sanitize_checkpoint_path(cfg):
    """Sanitize checkpoint filename to avoid Hydra parsing issues."""
    if not cfg.checkpoint or "=" not in cfg.checkpoint:
        return
    if Path(cfg.checkpoint).is_absolute() or "/" in cfg.checkpoint:
        return  # Skip for absolute paths or Hugging Face repo IDs
    
    sanitized_checkpoint = cfg.checkpoint.replace("=", "-")
    checkpoint_path = Path(cfg.train_folder) / cfg.checkpoint
    sanitized_path = Path(cfg.train_folder) / sanitized_checkpoint
    
    if sanitized_path.exists():
        print(f"Using sanitized checkpoint path: {sanitized_checkpoint}")
        cfg.checkpoint = sanitized_checkpoint
    elif checkpoint_path.exists():
        print(f"Warning: Checkpoint filename contains '=' which may cause Hydra parsing issues.")
        print(f"Consider renaming: {checkpoint_path} -> {sanitized_path}")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")


def _setup_wandb(cfg, log_dir):
    """Setup wandb logger."""
    import hashlib
    os.makedirs(log_dir / "wandb", exist_ok=True)
    checkpoint_stem = Path(cfg.checkpoint).stem
    
    # Get wandb config with fallbacks
    project = OmegaConf.select(cfg, "logger.project", default="mode_libero_eval")
    group = OmegaConf.select(cfg, "logger.group", default="mode_libero_eval")
    mode = OmegaConf.select(cfg, "logger.mode", default="online")
    run_id = OmegaConf.select(cfg, "logger.id", default=None)
    entity = OmegaConf.select(cfg, "logger.entity", default=None) or cfg.get("wandb_entity", None)
    
    # Clean up None/empty values
    entity = None if entity in ("null", "", None) else entity
    run_id = None if run_id in ("null", "") else run_id
    
    # Generate unique run_id if not provided
    if run_id is None:
        unique_str = f"{checkpoint_stem}_{time.time()}"
        run_id = hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    wandb.init(
        project=project,
        entity=entity,
        name=checkpoint_stem,
        group=group,
        config={
            "checkpoint": cfg.checkpoint,
            "benchmark_name": cfg.benchmark_name,
            "num_sequences": cfg.num_sequences,
            "n_eval": cfg.n_eval,
            "max_steps": cfg.max_steps,
        },
        dir=str(log_dir / "wandb"),
        mode=mode,
        id=run_id,
        resume="allow",
    )

@hydra.main(config_path="../../conf", config_name="mode_evaluate_libero_msillm")
def main(cfg: DictConfig):
    seed_everything(0, workers=True)
    
    # Handle checkpoint path
    _resolve_checkpoint_path(cfg)
    
    # Sanitize checkpoint filename if needed
    _sanitize_checkpoint_path(cfg)
    
    # Handle device selection
    device_id, device_str = _get_device_config(cfg)
    print(f"Using device: {device_str} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    
    # Load model using utility function (handles MS-ILLM automatically)
    use_ema = getattr(cfg, 'use_ema_weights', False)
    model, _, dm, _, loaded_cfg = get_msillm_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite if hasattr(cfg, 'eval_cfg_overwrite') else {},
        device_id=device_id,
        prep_dm_and_deps=False,
        use_ema_weights=use_ema
    )
    
    # Setup MS-ILLM BPP measurement and video settings
    _setup_msillm_features(model, cfg)
    
    # Store compress_gripper setting on model for video reconstruction
    # Try loaded_cfg first (merged config), then fallback to cfg
    compress_gripper = OmegaConf.select(loaded_cfg, "msillm.compress_gripper", 
                                       default=OmegaConf.select(cfg, "msillm.compress_gripper", default=True))
    model._compress_gripper = compress_gripper
    print(f"[Video] compress_gripper={compress_gripper} (stored on model)")
    
    # Store compress_rgb setting on model for video reconstruction
    # Try loaded_cfg first (merged config), then fallback to cfg
    compress_rgb = OmegaConf.select(loaded_cfg, "msillm.compress_rgb", 
                                    default=OmegaConf.select(cfg, "msillm.compress_rgb", default=True))
    model._compress_rgb = compress_rgb
    print(f"[Video] compress_rgb={compress_rgb} (stored on model)")
    
    # Ensure DataModule is setup to load statistics
    if not hasattr(dm, 'train_datasets') or not dm.train_datasets:
        dm.setup()
    
    model.eval()

    # Get log directory based on checkpoint name (without extension)
    # If using default pretrain_chk and MS-ILLM entrypoint is specified, use entrypoint as directory name
    log_checkpoint_name = cfg.checkpoint
    if cfg.checkpoint == "mbreuss/MoDE_LIBERO_10":  # Check if using default pretrain_chk
        msillm_entrypoint = OmegaConf.select(cfg, 'eval_cfg_overwrite.msillm.entrypoint', default=None)
        if msillm_entrypoint:
            log_checkpoint_name = msillm_entrypoint
            print(f"Using MS-ILLM entrypoint '{msillm_entrypoint}' for log directory")
    log_dir = get_log_dir(cfg.log_dir, checkpoint_name=log_checkpoint_name)
    
    # Load transforms (prefer validation transforms if available)
    transforms = _load_transforms(loaded_cfg, dm, cfg)

    print(f"[INFO] Loaded transforms: {transforms.keys() if hasattr(transforms, 'keys') else transforms}")

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
        device=device_str
    )

    # Setup wandb logger
    if cfg.log_wandb:
        _setup_wandb(cfg, log_dir)

    eval_libero.setup()
    eval_libero.start()

    if cfg.log_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
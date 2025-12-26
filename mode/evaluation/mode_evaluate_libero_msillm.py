import json
import logging
import os
from pathlib import Path
import sys
import time
import gc

import cv2
import hydra
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

from mode.evaluation.utils import get_msillm_mode_and_env
from mode.evaluation.multistep_sequences import get_sequences
from mode.utils.bpp_utils import (
    calculate_bpp_from_encoder_output,
    calculate_bpp_from_hyperprior_output,
    accumulate_bpp_stats,
    compute_average_bpp,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.utils import get_task_embs


log_print = logging.getLogger(__name__)


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
        # Get wrapped compression method for BPP measurement
        # First try to get from model._bpp_wrapper (set in main)
        compress_method = getattr(model, "_bpp_wrapper", None)
        
        # Fallback: try to find wrapper from msillm_model directly
        if compress_method is None:
            msillm_model = getattr(model, "msillm_model", None)
            if msillm_model is not None:
                if hasattr(msillm_model, "compress") and isinstance(msillm_model.compress, LatentCaptureWrapper):
                    compress_method = msillm_model.compress
                elif hasattr(msillm_model, "encoder") and hasattr(msillm_model.encoder, "forward") and isinstance(msillm_model.encoder.forward, LatentCaptureWrapper):
                    compress_method = msillm_model.encoder.forward
        
        if compress_method is None:
            log_print.warning(f"[BPP] No compress_method wrapper found for task {task_str} - BPP will not be measured")
        else:
            log_print.info(f"[BPP] Found compress_method wrapper for task {task_str}")
        
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
        init_states = torch.load(init_states_path, weights_only=False)
        num_success = 0
        pbar = tqdm(range(self.n_eval), desc=f"Evaluating {task_i.language[:30]}")
        for i in pbar:
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
                if compress_method is not None:
                    compress_method.clear()
                
                # Store flag for video saving before model.step
                save_frame_this_step = store_video_this_rollout and hasattr(model, '_store_reconstructed_frame')
                if save_frame_this_step:
                    model._save_frame_this_step = True
                
                actions = model.step(data, goal)
                
                # Calculate BPP from captured latents (rgb_static + rgb_gripper)
                if compress_method is not None and len(compress_method.latents) > 0:
                    bpp_dict = {}
                    sensors = [k for k in ("rgb_static", "rgb_gripper") if k in data.get("rgb_obs", {})]
                    # Map most recent latents to sensors in call order
                    latents = compress_method.latents[-len(sensors):]
                    for sensor_name, latent in zip(sensors, latents):
                        img = data["rgb_obs"][sensor_name]  # (1, C, H, W)
                        img_for_bpp = img.squeeze(0)  # (C, H, W)
                        try:
                            if hasattr(latent, "latent_strings"):
                                bpp = calculate_bpp_from_hyperprior_output(latent, img_for_bpp.shape)
                            else:
                                bpp = calculate_bpp_from_encoder_output(latent, img_for_bpp, bits_per_element=8)
                            bpp_dict[sensor_name] = bpp
                        except Exception as e:
                            log_print.warning(f"Failed to calculate BPP for {sensor_name}: {e}")
                    # Accumulate BPP statistics
                    if bpp_dict:
                        self.bpp_stats = accumulate_bpp_stats(bpp_dict, self.bpp_stats)
                
                actions = actions.cpu().numpy()
                obs, reward, done, info = env.step(actions)

                if store_video_this_rollout:
                    # Use reconstructed image if flag is set, otherwise use original (which is already numpy)
                    if (hasattr(model, '_store_reconstructed_frame') and 
                        model._store_reconstructed_frame and 
                        hasattr(model, '_last_reconstructed_frame_tensor')):
                        # Only convert GPU tensor to numpy when using reconstructed frames
                        # Note: This GPU->CPU transfer is necessary because:
                        # 1. PyTorch GPU tensors cannot be directly converted to numpy
                        # 2. cv2.VideoWriter only accepts numpy arrays
                        recon_frame = model._last_reconstructed_frame_tensor  # [C, H, W] tensor on GPU
                        rgb_recon_np = (recon_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        rgb_recon_np = np.rot90(rgb_recon_np, k=2, axes=(0, 1))  # Rotate 180 degrees
                        frame = rgb_recon_np[..., ::-1]  # RGB to BGR for cv2
                    else:
                        # Use original frame (already numpy, no GPU->CPU transfer needed)
                        frame = obs['agentview_image']
                        # Fix: Rotate 180 degrees and convert RGB to BGR for cv2
                        if isinstance(frame, np.ndarray):
                            frame = np.rot90(frame, k=2, axes=(0, 1))
                            frame = frame[..., ::-1]
                    video_frames.append(frame)

                if done:
                    break

            if store_video_this_rollout:
                for frame in video_frames:
                    video_writer.write(frame)
                video_writer.release()

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

@hydra.main(config_path="../../conf", config_name="mode_evaluate_libero_msillm")
def main(cfg: DictConfig):
    seed_everything(0, workers=True)
    
    # Handle checkpoint path from environment variable if provided
    checkpoint_env = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_env:
        print(f"Using checkpoint from environment variable: {checkpoint_env}")
        cfg.checkpoint = checkpoint_env
    
    # If checkpoint is not specified (null/empty), use pretrain_chk from config_libero_msillm.yaml (Hugging Face repo)
    if not cfg.checkpoint or cfg.checkpoint in ("", "null", None):
        # Load config_libero_msillm to get pretrain_chk
        try:
            if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.initialize("../../conf")
            base_cfg = hydra.compose(config_name="config_libero_msillm")
            if hasattr(base_cfg, "pretrain_chk") and base_cfg.pretrain_chk:
                cfg.checkpoint = base_cfg.pretrain_chk
                print(f"No checkpoint specified, using pretrained checkpoint from config_libero_msillm.yaml: {cfg.checkpoint}")
            else:
                raise ValueError("No checkpoint specified and pretrain_chk not found in config_libero_msillm.yaml")
        except Exception as e:
            print(f"Error: Could not load pretrain_chk from config: {e}")
            raise ValueError("No checkpoint specified. Please provide checkpoint path or ensure pretrain_chk is set in config_libero_msillm.yaml")
    
    # Sanitize checkpoint filename: replace '=' with '-' to avoid Hydra parsing issues
    # (Only if checkpoint is not a Hugging Face repo ID)
    if cfg.checkpoint and "=" in cfg.checkpoint and not Path(cfg.checkpoint).is_absolute() and "/" not in cfg.checkpoint:
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
    
    # Handle CUDA_VISIBLE_DEVICES properly for torch device selection
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        device_id = 0
        device_str = "cuda:0"
    else:
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
    
    # Wrap MS-ILLM compression method for BPP measurement
    # This captures latents during forward pass for BPP calculation
    # NOTE: We wrap AFTER get_msillm_mode_and_env because patch happens inside it
    # But wrapper will still work because utils.py uses getattr(msillm, "compress") for dynamic lookup
    msillm_model = getattr(model, "msillm_model", None)
    wrapper = None
    if msillm_model is not None:
        if hasattr(msillm_model, "compress"):
            # Store original compress method
            original_compress = msillm_model.compress
            # Wrap it
            wrapper = LatentCaptureWrapper(original_compress)
            msillm_model.compress = wrapper
            print(f"[BPP] Wrapped msillm_model.compress for BPP measurement")
        elif hasattr(msillm_model.encoder, "forward"):
            # Store original encoder forward method
            original_encoder_forward = msillm_model.encoder.forward
            # Wrap it
            wrapper = LatentCaptureWrapper(original_encoder_forward)
            msillm_model.encoder.forward = wrapper
            print(f"[BPP] Wrapped msillm_model.encoder.forward for BPP measurement")
    
    # Store wrapper on model object for easy access in evaluate_task
    if wrapper is not None:
        model._bpp_wrapper = wrapper
        print(f"[BPP] Stored wrapper on model._bpp_wrapper")
    else:
        print(f"[BPP] WARNING: No wrapper created - BPP measurement may not work")
    
    # Enable storing reconstructed frames for video (if MS-ILLM is used and config allows it)
    use_reconstructed = getattr(cfg, 'use_reconstructed_video', True)  # Default to True for backward compatibility
    if hasattr(model, 'msillm_model') and model.msillm_model is not None and use_reconstructed:
        model._store_reconstructed_frame = True
        print("[Video] Will save MS-ILLM reconstructed images to video")
    else:
        model._store_reconstructed_frame = False
        if hasattr(model, 'msillm_model') and model.msillm_model is not None:
            print("[Video] Will save original env images to video (use_reconstructed_video=False)")
        else:
            print("[Video] Will save original env images to video (no MS-ILLM model)")
    
    # Ensure DataModule is setup to load statistics
    if not hasattr(dm, 'train_datasets') or not dm.train_datasets:
        dm.setup()
    
    model.eval()

    # Get log directory based on checkpoint name (without extension)
    log_dir = get_log_dir(cfg.log_dir, checkpoint_name=cfg.checkpoint)
    
    # Load transforms (prefer validation transforms if available)
    transforms = {}
    try:
        transforms_cfg = None
        # Try from loaded config first
        if hasattr(loaded_cfg, 'datamodule') and hasattr(loaded_cfg.datamodule, 'transforms'):
             transforms_cfg = loaded_cfg.datamodule.transforms.val if "val" in loaded_cfg.datamodule.transforms else loaded_cfg.datamodule.transforms
        
        # Fallback to DM transforms if config lookup failed
        if transforms_cfg is None and hasattr(dm, 'transforms') and dm.transforms is not None:
            if 'val' in dm.transforms:
                transforms_cfg = dm.transforms.val
            else:
                transforms_cfg = dm.transforms
        
        # Final fallback to current cfg
        if transforms_cfg is None:
            transforms_cfg = cfg.datamodule.transforms.val if "val" in cfg.datamodule.transforms else cfg.datamodule.transforms
            
        # Instantiate transforms properly (handle ListConfig -> Compose)
        transforms = _instantiate_transforms(transforms_cfg)
        
    except Exception as e:
        print(f"[WARNING] Failed to load transforms from DM/Config: {e}")
        import traceback
        traceback.print_exc()
        # Final fallback to existing DM transforms if already instantiated
        if hasattr(dm, 'transforms'):
             transforms = dm.transforms
        else:
             raise e

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
        os.makedirs(log_dir / "wandb", exist_ok=True)
        checkpoint_stem = Path(cfg.checkpoint).stem
        wandb_config = {
            "checkpoint": cfg.checkpoint,
            "benchmark_name": cfg.benchmark_name,
            "num_sequences": cfg.num_sequences,
            "n_eval": cfg.n_eval,
            "max_steps": cfg.max_steps,
        }

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
        
        # Generate unique run_id if not provided to avoid HTTP 409 conflicts
        if run_id is None:
            import time
            import hashlib
            # Create unique run_id based on checkpoint name and timestamp
            unique_str = f"{checkpoint_stem}_{time.time()}"
            run_id = hashlib.md5(unique_str.encode()).hexdigest()[:16]

        run = wandb.init(
            project=project,
            entity=entity,
            name=checkpoint_stem,
            group=group,
            config=wandb_config,
            dir=str(log_dir / "wandb"),
            mode=mode,
            id=run_id,
            resume="allow",  # Allow resuming if run_id exists (though unlikely with unique IDs)
        )

    eval_libero.setup()
    eval_libero.start()

    if cfg.log_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

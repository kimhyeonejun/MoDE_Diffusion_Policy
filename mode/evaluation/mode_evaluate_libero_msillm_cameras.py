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

from mode.evaluation.utils import (
    get_msillm_mode_and_env,
    load_msillm_from_torchhub,
    get_device,
    move_model_to_device,
    reconstruct_frame_for_video_dual_msillm,
)
import torch.nn.functional as F
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

# Import common functions from mode_evaluate_libero_msillm
from mode.evaluation.mode_evaluate_libero_msillm import (
    LatentCaptureWrapper,
    get_log_dir,
    _prepare_video_frame,
    _instantiate_transforms,
    _load_transforms,
    _get_device_config,
    _resolve_checkpoint_path,
    _sanitize_checkpoint_path,
    _setup_wandb,
    EvaluateLibero as BaseEvaluateLibero,
)

log_print = logging.getLogger(__name__)


def _get_bpp_wrapper(model, sensor_name="rgb_static"):
    """Helper to get BPP wrapper from model for a specific sensor."""
    # For dual MS-ILLM setup, wrappers are stored per sensor
    wrapper_attr = f"_bpp_wrapper_{sensor_name}"
    wrapper = getattr(model, wrapper_attr, None)
    return wrapper


def _calculate_bpp_from_latents(model, data, sensors):
    """Calculate BPP from captured latents for each sensor (supports dual MS-ILLM)."""
    bpp_dict = {}
    for sensor_name in sensors:
        wrapper = _get_bpp_wrapper(model, sensor_name)
        if wrapper is None or len(wrapper.latents) == 0:
            continue
        # Get the last latent for this sensor
        latent = wrapper.latents[-1] if len(wrapper.latents) > 0 else None
        if latent is None:
            continue
        img = data["rgb_obs"][sensor_name].squeeze(0)  # (C, H, W)
        bpp = calculate_bpp_from_hyperprior_output(latent, img.shape)
        bpp_dict[sensor_name] = bpp
    return bpp_dict


# _prepare_video_frame, LatentCaptureWrapper, get_log_dir are imported from mode_evaluate_libero_msillm


class EvaluateLibero(BaseEvaluateLibero):
    # __init__, setup, start, evaluate_policy are inherited from BaseEvaluateLibero
    # Only evaluate_task is overridden for dual MS-ILLM support

    def evaluate_task(self, model, task_i, task_emb, task_str, idx, sim_states=None, store_video=0):
        # Check for BPP wrappers (supports dual MS-ILLM)
        bpp_wrapper_static = _get_bpp_wrapper(model, "rgb_static")
        bpp_wrapper_gripper = _get_bpp_wrapper(model, "rgb_gripper")
        if bpp_wrapper_static is None and bpp_wrapper_gripper is None:
            log_print.warning(f"[BPP] No BPP wrapper found for task {task_str} - BPP will not be measured")
        else:
            log_print.info(f"[BPP] Found BPP wrappers for task {task_str} (static: {bpp_wrapper_static is not None}, gripper: {bpp_wrapper_gripper is not None})")
        
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
                
                # Clear captured latents for this step (supports dual MS-ILLM)
                if bpp_wrapper_static is not None:
                    bpp_wrapper_static.clear()
                if bpp_wrapper_gripper is not None:
                    bpp_wrapper_gripper.clear()
                
                actions = model.step(data, goal)
                
                # CRITICAL: model.step() uses action chunking, so embed_visual_obs is NOT called every step.
                # We need to manually reconstruct frames for video every step using the current data.
                if (store_video_this_rollout 
                    and hasattr(model, '_store_reconstructed_frame') 
                    and model._store_reconstructed_frame):
                    # Reconstruct rgb_static if available and compress_rgb is enabled
                    if "rgb_static" in data.get("rgb_obs", {}) and compress_rgb:
                        recon_frame = reconstruct_frame_for_video_dual_msillm(model, data["rgb_obs"]["rgb_static"], "rgb_static")
                        if recon_frame is not None:
                            model._last_reconstructed_frame_tensor_rgb_static = recon_frame.detach().clone()
                    
                    # Reconstruct rgb_gripper if available and compress_gripper is enabled
                    if "rgb_gripper" in data.get("rgb_obs", {}) and compress_gripper:
                        recon_frame_gripper = reconstruct_frame_for_video_dual_msillm(model, data["rgb_obs"]["rgb_gripper"], "rgb_gripper")
                        if recon_frame_gripper is not None:
                            model._last_reconstructed_frame_tensor_rgb_gripper = recon_frame_gripper.detach().clone()

                # Calculate BPP from captured latents (supports dual MS-ILLM)
                sensors = []
                if compress_rgb and "rgb_static" in data.get("rgb_obs", {}):
                    sensors.append("rgb_static")
                if compress_gripper and "rgb_gripper" in data.get("rgb_obs", {}):
                    sensors.append("rgb_gripper")
                bpp_dict = _calculate_bpp_from_latents(model, data, sensors)
                if bpp_dict:
                    self.bpp_stats = accumulate_bpp_stats(bpp_dict, self.bpp_stats)
                
                # CRITICAL: Prepare video frames BEFORE env.step() using the current data and stored reconstructed frames
                # This ensures we use the reconstructed frames we just created, not the next step's data
                if store_video_this_rollout:
                    store_reconstructed = (hasattr(model, '_store_reconstructed_frame') 
                                         and model._store_reconstructed_frame)
                    
                    # Save static frame (use reconstructed frame if available, otherwise use data)
                    frame_static = _prepare_video_frame(model, obs, store_reconstructed, sensor_name='rgb_static', data=data)
                    if frame_static is None:
                        # Fallback: use data if available
                        if data is not None and "rgb_static" in data.get("rgb_obs", {}):
                            img_tensor = data["rgb_obs"]["rgb_static"].squeeze(0)[0]  # [1, T, C, H, W] -> [C, H, W]
                            frame_static = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            frame_static = np.rot90(frame_static, k=2, axes=(0, 1))
                            frame_static = frame_static[..., ::-1]  # RGB to BGR
                    video_frames_static.append(frame_static)
                    
                    # Save gripper frame (use reconstructed frame if available, otherwise use data)
                    frame_gripper = _prepare_video_frame(model, obs, store_reconstructed, sensor_name='rgb_gripper', data=data)
                    if frame_gripper is None:
                        # Fallback: use data if available
                        if data is not None and "rgb_gripper" in data.get("rgb_obs", {}):
                            img_tensor = data["rgb_obs"]["rgb_gripper"].squeeze(0)[0]  # [1, T, C, H, W] -> [C, H, W]
                            frame_gripper = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            frame_gripper = np.rot90(frame_gripper, k=2, axes=(0, 1))
                            frame_gripper = frame_gripper[..., ::-1]  # RGB to BGR
                    video_frames_gripper.append(frame_gripper)
                
                actions = actions.cpu().numpy()
                obs, reward, done, info = env.step(actions)

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

    # create_cfg_for_libero, translate_obs_space, apply_transforms, process_env_obs are inherited from BaseEvaluateLibero


def _load_msillm_from_checkpoint(ckpt_path, device):
    """Load MS-ILLM model from a .ckpt checkpoint file."""
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MS-ILLM checkpoint not found: {ckpt_path}")
    
    if not ckpt_path.suffix == ".ckpt":
        raise ValueError(f"MS-ILLM checkpoint must be a .ckpt file, got: {ckpt_path}")
    
    print(f"[MS-ILLM] Loading from checkpoint: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # First, load model structure from torchhub (we need the model class)
    # Try to get entrypoint from checkpoint metadata, or use default
    hub_repo = "facebookresearch/NeuralCompression:main"
    entrypoint = "msillm_quality_1"  # Default
    
    # Try to infer entrypoint from checkpoint if available
    if 'hyperparameters' in checkpoint:
        hparams = checkpoint['hyperparameters']
        if isinstance(hparams, dict) and 'msillm_entrypoint' in hparams:
            entrypoint = hparams['msillm_entrypoint']
        elif hasattr(hparams, 'msillm_entrypoint'):
            entrypoint = getattr(hparams, 'msillm_entrypoint')
    
    # Load model structure from torchhub
    try:
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=False, verbose=False)
    except TypeError:
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=False)
    
    # Load weights from checkpoint
    # Check if checkpoint contains MS-ILLM weights
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Extract MS-ILLM weights (keys starting with 'msillm_model.')
    msillm_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('msillm_model.'):
            # Remove 'msillm_model.' prefix
            new_key = key[len('msillm_model.'):]
            msillm_state_dict[new_key] = value
    
    if len(msillm_state_dict) == 0:
        # If no msillm_model prefix, try to find MS-ILLM keys directly
        # This might be a standalone MS-ILLM checkpoint
        msillm_state_dict = state_dict
    
    if len(msillm_state_dict) > 0:
        msillm_model.load_state_dict(msillm_state_dict, strict=False)
        print(f"[MS-ILLM] Loaded {len(msillm_state_dict)} parameters from checkpoint")
    else:
        print(f"[MS-ILLM] Warning: No MS-ILLM weights found in checkpoint, using pretrained weights")
        # Fallback to pretrained weights
        try:
            msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=True, verbose=False)
        except TypeError:
            msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=True)
    
    msillm_model = move_model_to_device(msillm_model, device)
    msillm_model.eval()
    
    # Set MS-ILLM to compression mode (moves entropy bottlenecks to CPU)
    if hasattr(msillm_model, "update_tensor_devices"):
        try:
            msillm_model.update_tensor_devices("compress")
            print(f"[MS-ILLM] Set model to compression mode (partial-GPU)")
        except Exception as e:
            print(f"[MS-ILLM] Warning: Failed to update tensor devices for compression: {e}")
            # Try to set device_setting directly
            if hasattr(msillm_model, "_device_setting"):
                msillm_model._device_setting = "compress"
    
    return msillm_model


def _load_dual_msillm_models(cfg, device_id):
    """Load two separate MS-ILLM models for rgb_static and rgb_gripper.
    
    Supports both torchhub entrypoint and .ckpt checkpoint file paths.
    """
    from mode.evaluation.utils import get_device
    
    device = get_device(device_id)
    
    # Load MS-ILLM for rgb_static
    msillm_static = None
    if hasattr(cfg, 'msillm_static') and cfg.msillm_static is not None:
        msillm_cfg = cfg.msillm_static
        
        # Check if checkpoint path is specified
        if hasattr(msillm_cfg, 'checkpoint') and msillm_cfg.checkpoint:
            ckpt_path = msillm_cfg.checkpoint
            msillm_static = _load_msillm_from_checkpoint(ckpt_path, device)
            print(f"[MS-ILLM] Loaded MS-ILLM for rgb_static from checkpoint: {ckpt_path}")
        else:
            # Use torchhub entrypoint
            msillm_static, _ = load_msillm_from_torchhub(OmegaConf.create({"msillm": msillm_cfg}))
            if msillm_static is not None:
                msillm_static = move_model_to_device(msillm_static, device)
                msillm_static.eval()
                # Set to compression mode
                if hasattr(msillm_static, "update_tensor_devices"):
                    try:
                        msillm_static.update_tensor_devices("compress")
                    except Exception as e:
                        print(f"[MS-ILLM] Warning: Failed to update tensor devices for rgb_static: {e}")
                        if hasattr(msillm_static, "_device_setting"):
                            msillm_static._device_setting = "compress"
                entrypoint = msillm_cfg.get('entrypoint', 'default')
                print(f"[MS-ILLM] Loaded MS-ILLM for rgb_static from torchhub: {entrypoint}")
    
    # Load MS-ILLM for rgb_gripper
    msillm_gripper = None
    if hasattr(cfg, 'msillm_gripper') and cfg.msillm_gripper is not None:
        msillm_cfg = cfg.msillm_gripper
        
        # Check if checkpoint path is specified
        if hasattr(msillm_cfg, 'checkpoint') and msillm_cfg.checkpoint:
            ckpt_path = msillm_cfg.checkpoint
            msillm_gripper = _load_msillm_from_checkpoint(ckpt_path, device)
            print(f"[MS-ILLM] Loaded MS-ILLM for rgb_gripper from checkpoint: {ckpt_path}")
        else:
            # Use torchhub entrypoint
            msillm_gripper, _ = load_msillm_from_torchhub(OmegaConf.create({"msillm": msillm_cfg}))
            if msillm_gripper is not None:
                msillm_gripper = move_model_to_device(msillm_gripper, device)
                msillm_gripper.eval()
                # Set to compression mode
                if hasattr(msillm_gripper, "update_tensor_devices"):
                    try:
                        msillm_gripper.update_tensor_devices("compress")
                    except Exception as e:
                        print(f"[MS-ILLM] Warning: Failed to update tensor devices for rgb_gripper: {e}")
                        if hasattr(msillm_gripper, "_device_setting"):
                            msillm_gripper._device_setting = "compress"
                entrypoint = msillm_cfg.get('entrypoint', 'default')
                print(f"[MS-ILLM] Loaded MS-ILLM for rgb_gripper from torchhub: {entrypoint}")
    
    return msillm_static, msillm_gripper


def _patch_embed_visual_obs_for_dual_msillm(model, msillm_static, msillm_gripper, compress_gripper=True, compress_rgb=True):
    """Patch embed_visual_obs to use different MS-ILLM models for each camera."""
    import types
    from mode.evaluation.utils import _clip_mean_std
    import torch.nn.functional as F
    
    # Ensure MS-ILLM models are in compression mode
    for msillm_model, name in [(msillm_static, "rgb_static"), (msillm_gripper, "rgb_gripper")]:
        if msillm_model is not None:
            if hasattr(msillm_model, "update_tensor_devices"):
                try:
                    msillm_model.update_tensor_devices("compress")
                    print(f"[MS-ILLM] Set {name} model to compression mode (partial-GPU)")
                except Exception as e:
                    print(f"[MS-ILLM] Warning: Failed to update tensor devices for {name}: {e}")
                    # Try to set device_setting directly
                    if hasattr(msillm_model, "_device_setting"):
                        msillm_model._device_setting = "compress"
    
    orig = getattr(model, "embed_visual_obs", None)
    if orig is None or not callable(orig):
        return
    
    def _reconstruct_normed(x01: torch.Tensor, msillm_model, sensor_name: str = "rgb_static"):
        """Reconstruct and normalize using a specific MS-ILLM model."""
        if msillm_model is None:
            # Fallback: just normalize
            mean, std = _clip_mean_std(x01.device, x01.dtype)
            return (x01 - mean) / std, None
        
        mean, std = _clip_mean_std(x01.device, x01.dtype)
        b, t, c, h, w = x01.shape
        x01_bt = x01.reshape(b * t, c, h, w)
        
        # MS-ILLM requires images to be divisible by 64
        factor = 64
        resize_needed = False
        if h % factor != 0 or w % factor != 0:
            new_h = ((h + factor - 1) // factor) * factor
            new_w = ((w + factor - 1) // factor) * factor
            x01_bt_resized = F.interpolate(x01_bt, size=(new_h, new_w), mode='bilinear', align_corners=False)
            resize_needed = True
        else:
            x01_bt_resized = x01_bt
        
        with torch.no_grad():
            compress_method = getattr(msillm_model, "compress", None)
            if compress_method is not None:
                compressed = compress_method(x01_bt_resized, force_cpu=False)
                recon_resized = msillm_model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
            else:
                recon_resized = x01_bt_resized
        
        # Resize back to original size if needed
        if resize_needed:
            recon = F.interpolate(recon_resized, size=(h, w), mode='bilinear', align_corners=False)
        else:
            recon = recon_resized
        
        recon = recon.reshape(b, t, c, h, w)
        out = (recon - mean) / std
        return out, recon
    
    def _patched(self, rgb_static, rgb_gripper, latent_goal):
        # Reconstruct rgb_static using msillm_static
        if compress_rgb:
            rgb_static_recon, rgb_static_recon_denorm = _reconstruct_normed(rgb_static, msillm_static, "rgb_static")
        else:
            mean, std = _clip_mean_std(rgb_static.device, rgb_static.dtype)
            rgb_static_recon = (rgb_static - mean) / std
            rgb_static_recon_denorm = None
        
        # Reconstruct rgb_gripper using msillm_gripper
        if compress_gripper:
            rgb_gripper_recon, rgb_gripper_recon_denorm = _reconstruct_normed(rgb_gripper, msillm_gripper, "rgb_gripper")
        else:
            mean, std = _clip_mean_std(rgb_gripper.device, rgb_gripper.dtype)
            rgb_gripper_recon = (rgb_gripper - mean) / std
            rgb_gripper_recon_denorm = None
        
        # Store reconstructed frames for video if enabled
        # Always store if _store_reconstructed_frame is True, regardless of compress_rgb/compress_gripper flags
        # because the flags only control whether we reconstruct, but if we reconstructed, we should store it
        if hasattr(self, '_store_reconstructed_frame') and self._store_reconstructed_frame:
            if compress_rgb and rgb_static_recon_denorm is not None:
                # Extract single frame: [C, H, W] from [B, T, C, H, W]
                if rgb_static_recon_denorm.dim() == 5:
                    static_frame = rgb_static_recon_denorm[0, 0]  # [B, T, C, H, W] -> [C, H, W]
                elif rgb_static_recon_denorm.dim() == 4:
                    static_frame = rgb_static_recon_denorm[0]  # [B, C, H, W] -> [C, H, W]
                else:
                    static_frame = rgb_static_recon_denorm  # Already [C, H, W]
                # Detach and clone to avoid gradient issues and ensure persistence
                self._last_reconstructed_frame_tensor_rgb_static = static_frame.detach().clone()
            
            if compress_gripper and rgb_gripper_recon_denorm is not None:
                # Extract single frame: [C, H, W] from [B, T, C, H, W]
                if rgb_gripper_recon_denorm.dim() == 5:
                    gripper_frame = rgb_gripper_recon_denorm[0, 0]  # [B, T, C, H, W] -> [C, H, W]
                elif rgb_gripper_recon_denorm.dim() == 4:
                    gripper_frame = rgb_gripper_recon_denorm[0]  # [B, C, H, W] -> [C, H, W]
                else:
                    gripper_frame = rgb_gripper_recon_denorm  # Already [C, H, W]
                # Detach and clone to avoid gradient issues and ensure persistence
                self._last_reconstructed_frame_tensor_rgb_gripper = gripper_frame.detach().clone()
        
        return orig(rgb_static_recon, rgb_gripper_recon, latent_goal)
    
    model.embed_visual_obs = types.MethodType(_patched, model)
    print("[MS-ILLM] Patched embed_visual_obs for dual MS-ILLM setup")


def _setup_msillm_features(model, cfg):
    """Setup MS-ILLM BPP measurement wrapper and video settings (supports dual MS-ILLM)."""
    # Check if we have dual MS-ILLM setup
    has_msillm_static = hasattr(model, 'msillm_model_rgb_static') and model.msillm_model_rgb_static is not None
    has_msillm_gripper = hasattr(model, 'msillm_model_rgb_gripper') and model.msillm_model_rgb_gripper is not None
    has_msillm_single = hasattr(model, 'msillm_model') and model.msillm_model is not None
    
    # Wrap MS-ILLM compression methods for BPP measurement
    if has_msillm_static:
        msillm_static = model.msillm_model_rgb_static
        if hasattr(msillm_static, "compress"):
            wrapper_static = LatentCaptureWrapper(msillm_static.compress)
            msillm_static.compress = wrapper_static
            model._bpp_wrapper_rgb_static = wrapper_static
            print(f"[BPP] Wrapped msillm_model_rgb_static.compress for BPP measurement")
    
    if has_msillm_gripper:
        msillm_gripper = model.msillm_model_rgb_gripper
        if hasattr(msillm_gripper, "compress"):
            wrapper_gripper = LatentCaptureWrapper(msillm_gripper.compress)
            msillm_gripper.compress = wrapper_gripper
            model._bpp_wrapper_rgb_gripper = wrapper_gripper
            print(f"[BPP] Wrapped msillm_model_rgb_gripper.compress for BPP measurement")
    
    # Fallback to single MS-ILLM wrapper for backward compatibility
    if has_msillm_single and not (has_msillm_static or has_msillm_gripper):
        msillm_model = model.msillm_model
        if hasattr(msillm_model, "compress"):
            wrapper = LatentCaptureWrapper(msillm_model.compress)
            msillm_model.compress = wrapper
            model._bpp_wrapper = wrapper
            print(f"[BPP] Wrapped msillm_model.compress for BPP measurement")
    
    # Enable storing reconstructed frames for video
    use_reconstructed = getattr(cfg, 'use_reconstructed_video', True)
    has_any_msillm = has_msillm_static or has_msillm_gripper or has_msillm_single
    
    if has_any_msillm and use_reconstructed:
        model._store_reconstructed_frame = True
        print("[Video] Will save MS-ILLM reconstructed images to video")
    else:
        model._store_reconstructed_frame = False
        if has_any_msillm:
            print("[Video] Will save original env images to video (use_reconstructed_video=False)")
        else:
            print("[Video] Will save original env images to video (no MS-ILLM model)")


# _resolve_checkpoint_path, _sanitize_checkpoint_path, _setup_wandb are imported from mode_evaluate_libero_msillm

@hydra.main(config_path="../../conf", config_name="mode_evaluate_libero_msillm_camera")
def main(cfg: DictConfig):
    seed_everything(0, workers=True)
    
    # Handle checkpoint path
    _resolve_checkpoint_path(cfg)
    
    # Sanitize checkpoint filename if needed
    _sanitize_checkpoint_path(cfg)
    
    # Handle device selection
    device_id, device_str = _get_device_config(cfg)
    print(f"Using device: {device_str} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    
    # Check if we're using dual MS-ILLM setup
    use_dual_msillm = (hasattr(cfg, 'msillm_static') and cfg.msillm_static is not None) or \
                      (hasattr(cfg, 'msillm_gripper') and cfg.msillm_gripper is not None)
    
    if use_dual_msillm:
        # Load model without MS-ILLM (we'll add our own)
        use_ema = getattr(cfg, 'use_ema_weights', False)
        # Temporarily disable MS-ILLM in eval_cfg_overwrite to avoid loading single MS-ILLM
        eval_cfg_overwrite = cfg.eval_cfg_overwrite if hasattr(cfg, 'eval_cfg_overwrite') else {}
        eval_cfg_overwrite_no_msillm = {**eval_cfg_overwrite}
        if 'msillm' in eval_cfg_overwrite_no_msillm:
            del eval_cfg_overwrite_no_msillm['msillm']
        
        model, _, dm, _, loaded_cfg = get_msillm_mode_and_env(
            cfg.train_folder,
            cfg.dataset_path,
            cfg.checkpoint,
            env=None,
            lang_embeddings=None,
            eval_cfg_overwrite=eval_cfg_overwrite_no_msillm,
            device_id=device_id,
            prep_dm_and_deps=False,
            use_ema_weights=use_ema
        )
        
        # Load dual MS-ILLM models
        msillm_static, msillm_gripper = _load_dual_msillm_models(cfg, device_id)
        
        # Attach MS-ILLM models to model
        if msillm_static is not None:
            model.msillm_model_rgb_static = msillm_static
        if msillm_gripper is not None:
            model.msillm_model_rgb_gripper = msillm_gripper
        
        # Patch embed_visual_obs to use dual MS-ILLM
        compress_gripper = OmegaConf.select(loaded_cfg, "msillm.compress_gripper", 
                                           default=OmegaConf.select(cfg, "msillm.compress_gripper", default=True))
        compress_rgb = OmegaConf.select(loaded_cfg, "msillm.compress_rgb", 
                                        default=OmegaConf.select(cfg, "msillm.compress_rgb", default=True))
        _patch_embed_visual_obs_for_dual_msillm(model, msillm_static, msillm_gripper, compress_gripper, compress_rgb)
        
        # Setup MS-ILLM BPP measurement and video settings
        _setup_msillm_features(model, cfg)
        
        model._compress_gripper = compress_gripper
        model._compress_rgb = compress_rgb
        print(f"[Video] compress_gripper={compress_gripper}, compress_rgb={compress_rgb} (stored on model)")
        
        # Ensure reconstructed frames are stored for video (dual MS-ILLM mode)
        use_reconstructed = getattr(cfg, 'use_reconstructed_video', True)
        if use_reconstructed and (msillm_static is not None or msillm_gripper is not None):
            model._store_reconstructed_frame = True
            print("[Video] Will save MS-ILLM reconstructed images to video (dual MS-ILLM mode)")
        else:
            model._store_reconstructed_frame = False
            print("[Video] Will save original env images to video (use_reconstructed_video=False or no MS-ILLM)")
    else:
        # Use single MS-ILLM (original behavior)
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
        
        # Store compress_gripper and compress_rgb settings on model for video reconstruction
        compress_gripper = OmegaConf.select(loaded_cfg, "msillm.compress_gripper", 
                                           default=OmegaConf.select(cfg, "msillm.compress_gripper", default=True))
        compress_rgb = OmegaConf.select(loaded_cfg, "msillm.compress_rgb", 
                                        default=OmegaConf.select(cfg, "msillm.compress_rgb", default=True))
        model._compress_gripper = compress_gripper
        model._compress_rgb = compress_rgb
        print(f"[Video] compress_gripper={compress_gripper}, compress_rgb={compress_rgb} (stored on model)")
    
    # Ensure DataModule is setup to load statistics
    if not hasattr(dm, 'train_datasets') or not dm.train_datasets:
        dm.setup()
    
    model.eval()

    # Get log directory based on checkpoint name (without extension)
    # For dual MS-ILLM, include both checkpoint names in directory
    log_checkpoint_name = cfg.checkpoint
    if use_dual_msillm:
        # Build directory name from dual MS-ILLM checkpoints
        msillm_names = []
        
        # Get static MS-ILLM name
        if hasattr(cfg, 'msillm_static') and cfg.msillm_static is not None:
            msillm_cfg_static = cfg.msillm_static
            if hasattr(msillm_cfg_static, 'checkpoint') and msillm_cfg_static.checkpoint:
                # Extract name from checkpoint path
                ckpt_path_static = Path(msillm_cfg_static.checkpoint)
                msillm_names.append(f"static_{ckpt_path_static.stem}")
            elif hasattr(msillm_cfg_static, 'entrypoint'):
                msillm_names.append(f"static_{msillm_cfg_static.entrypoint}")
        
        # Get gripper MS-ILLM name
        if hasattr(cfg, 'msillm_gripper') and cfg.msillm_gripper is not None:
            msillm_cfg_gripper = cfg.msillm_gripper
            if hasattr(msillm_cfg_gripper, 'checkpoint') and msillm_cfg_gripper.checkpoint:
                # Extract name from checkpoint path
                ckpt_path_gripper = Path(msillm_cfg_gripper.checkpoint)
                msillm_names.append(f"gripper_{ckpt_path_gripper.stem}")
            elif hasattr(msillm_cfg_gripper, 'entrypoint'):
                msillm_names.append(f"gripper_{msillm_cfg_gripper.entrypoint}")
        
        if msillm_names:
            log_checkpoint_name = "__".join(msillm_names)
            print(f"Using dual MS-ILLM directory name: {log_checkpoint_name}")
        else:
            # Fallback to main checkpoint name
            if cfg.checkpoint == "mbreuss/MoDE_LIBERO_10":
                msillm_entrypoint = OmegaConf.select(cfg, 'eval_cfg_overwrite.msillm.entrypoint', default=None)
                if msillm_entrypoint:
                    log_checkpoint_name = msillm_entrypoint
    elif cfg.checkpoint == "mbreuss/MoDE_LIBERO_10":  # Check if using default pretrain_chk
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
import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os 
from typing import Optional, Tuple
import wandb
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import types
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import importlib

# This is for using the locally installed repo clone when using slurm
repo_root = Path(__file__).absolute().parents[1]
sys.path.insert(0, repo_root.as_posix())

# Add LIBERO submodule to path so 'libero' module can be imported
libero_repo_dir = repo_root / "LIBERO"
if libero_repo_dir.exists():
    sys.path.insert(0, str(libero_repo_dir))
    # Also set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{libero_repo_dir}:{current_pythonpath}" if current_pythonpath else str(libero_repo_dir)

# Add local SAM3 to path (prioritize local over site-packages).
# This is required for the SAM3-guided reconstruction loss hook below.
local_sam3_dir = repo_root / "sam" / "sam3"
if local_sam3_dir.exists():
    sys.path.insert(0, str(local_sam3_dir))

# Import prompt helper and SAM3 utilities from our modularized utils
# Must be imported AFTER path setup
from sam.utils.prompts import build_prompt_candidates
from sam.utils.sam3_weight_map import get_sam3_processor, compute_weight_map

from mode.utils.utils import get_last_checkpoint, initialize_pretrained_weights, print_system_env_info
from mode.training_utils import (
    patch_on_before_zero_grad_for_msillm,
    set_requires_grad,
    count_params,
    cfg_get,
    patch_optimizer_to_only_train_selected,
    freeze_all_except_vision_encoders,
    clear_cuda_cache,
    log_rank_0,
    get_msillm_identifier,
    extract_msillm_identifier_from_checkpoint_path,
    setup_callbacks,
    setup_logger,
    extract_compression_modules,
    load_pretrained_weights_from_hf,
    load_msillm_from_torchhub,
    clip_mean_std,
    cleanup_distributed,
)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global custom-loss config cache (populated in `train()` so loss hooks can read cfg values).
_CUSTOM_LOSS_CFG = None

def _load_custom_loss_hook(cfg: DictConfig):
    """
    Load a user-provided custom loss hook without modifying core model code.

    Enable via either:
      - Hydra: `custom_loss.fn=<module.path:function>` (and optional `custom_loss.weight`, `custom_loss.log_name`)
      - Env var: `CUSTOM_LOSS_FN=<module.path:function>`

    The hook signature should be:

        def custom_loss(model: LightningModule, batch, batch_idx: int) -> torch.Tensor | float:
            ...
            return loss
    """
    fn_path = os.environ.get("CUSTOM_LOSS_FN")
    weight = 1.0
    log_name = "train/custom_loss"

    if "custom_loss" in cfg:
        try:
            fn_path = cfg.custom_loss.get("fn", fn_path)
            weight = float(cfg.custom_loss.get("weight", weight))
            log_name = str(cfg.custom_loss.get("log_name", log_name))
        except Exception:
            # DictConfig access can vary depending on struct mode
            pass

    if not fn_path:
        return None, 0.0, log_name

    if ":" in fn_path:
        module_name, attr = fn_path.split(":", 1)
    elif "." in fn_path:
        module_name, attr = fn_path.rsplit(".", 1)
    else:
        raise ValueError(
            f"Invalid custom loss hook path: {fn_path}. "
            f"Use '<module>.<fn>' or '<module>:<fn>'."
        )

    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"custom_loss.fn must be callable, got: {type(fn).__name__}")

    return fn, weight, log_name


def _patch_training_step_with_custom_loss(
    model: LightningModule,
    *,
    loss_fn,
    weight: float,
    log_name: str,
) -> None:
    """
    Monkey-patch model.training_step so you can add arbitrary custom loss terms.
    This keeps changes localized to this training entrypoint.
    """
    if loss_fn is None:
        return

    orig_training_step = getattr(model, "training_step", None)
    if orig_training_step is None or not callable(orig_training_step):
        raise AttributeError("Model has no callable training_step to patch")

    def _patched_training_step(batch, batch_idx: int):  # type: ignore
        base_loss = orig_training_step(batch, batch_idx)

        try:
            extra = loss_fn(model=model, batch=batch, batch_idx=batch_idx)
            if not torch.is_tensor(extra):
                extra = torch.tensor(float(extra), device=base_loss.device, dtype=base_loss.dtype)
            else:
                extra = extra.to(device=base_loss.device, dtype=base_loss.dtype)
        except Exception as e:
            # Fail fast with a clear message; custom losses are user code.
            raise RuntimeError(f"Custom loss hook failed ({getattr(loss_fn, '__name__', 'custom_loss')}): {e}") from e

        total = base_loss + (extra * float(weight))

        # Log (on_step=False so it aggregates like the base training logs in MoDEAgent)
        try:
            model.log(log_name, extra, on_step=False, on_epoch=True, sync_dist=True)
        except Exception:
            # Don't crash training if logging backend differs
            pass

        return total

    # type: ignore[method-assign]
    model.training_step = _patched_training_step
    log_rank_0(f"Patched training_step with custom loss: {loss_fn} (weight={weight}, log={log_name})")

# SAM3-related helper functions are now in sam.utils.sam3_weight_map

def sam3_weighted_recon_loss(model: LightningModule, batch, batch_idx: int) -> torch.Tensor:
    """
    SAM3-guided reconstruction loss:

      L_recon = || ((I_pred - I_gt) * (1 + alpha * M_sam)) ||^2

    - I_gt: ground-truth image in [0,1]
    - I_pred: MS-ILLM reconstruction in [0,1] (decoder output, keeps grads)
    - M_sam: binary mask from SAM3 on I_gt (no grads)

    Notes:
    - We compute M_sam from the FIRST timestep only and broadcast over T.
    - Prompts are derived from dataset_batch['lang_text'] (object phrases).
    """
    cache = getattr(model, "_msillm_last_recon_cache", None)
    if cache is None:
        # If forward patch didn't populate anything, skip.
        return torch.tensor(0.0, device=model.device)

    cfg = _CUSTOM_LOSS_CFG
    alpha = float(cfg_get(cfg, "sam3_alpha", 1.0))
    view = str(cfg_get(cfg, "sam3_view", "both"))  # "static", "gripper", "both"
    conf_thr = float(cfg_get(cfg, "sam3_confidence_threshold", 0.05))
    thresholds = cfg_get(cfg, "sam3_thresholds", [conf_thr, 0.10, 0.05])
    if isinstance(thresholds, (float, int)):
        thresholds = [float(thresholds)]
    thresholds = [float(t) for t in thresholds]
    max_samples = cfg_get(cfg, "sam3_max_samples", None)  # Limit SAM3 processing for performance
    if max_samples is not None:
        max_samples = int(max_samples)

    # Extract lang_text from batch (use first sample's text for prompts)
    # Batch can be dict of datasets (train) or a single dataset batch (val-like)
    if isinstance(batch, dict) and "rgb_obs" not in batch:
        # Multiple datasets: use first dataset's lang_text
        dataset_batch = list(batch.values())[0]
    else:
        dataset_batch = batch
    
    lang_text = dataset_batch.get("lang_text", None)
    if isinstance(lang_text, (list, tuple)):
        lang_text = lang_text[0] if len(lang_text) > 0 else ""
    if torch.is_tensor(lang_text):
        # Unlikely, but keep safe.
        lang_text = ""
    prompts = build_prompt_candidates(lang_text or "")
    if not prompts:
        prompts = ["object"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sam3_processor = get_sam3_processor(device=device, logger=logger)

    # Pull gt/recon from cache populated by forward patch.
    # Shapes are expected to be (B,T,C,H,W) in [0,1].
    # Cache contains batch-level data, so we process it directly.
    total = None
    count = 0

    if view in ("static", "both") and "rgb_static_gt" in cache and "rgb_static_recon" in cache:
        gt = cache["rgb_static_gt"]
        pred = cache["rgb_static_recon"]
        wm = compute_weight_map(
            gt,
            sam3_processor,
            conf_thr,
            prompts,
            thresholds,
            alpha,
            max_samples=max_samples,
        )
        diff = (pred - gt) * wm
        loss = (diff * diff).mean()
        total = loss if total is None else (total + loss)
        count += 1

    if view in ("gripper", "both") and "rgb_gripper_gt" in cache and "rgb_gripper_recon" in cache:
        gt = cache["rgb_gripper_gt"]
        pred = cache["rgb_gripper_recon"]
        wm = compute_weight_map(
            gt,
            sam3_processor,
            conf_thr,
            prompts,
            thresholds,
            alpha,
            max_samples=max_samples,
        )
        diff = (pred - gt) * wm
        loss = (diff * diff).mean()
        total = loss if total is None else (total + loss)
        count += 1

    if total is None:
        return torch.tensor(0.0, device=model.device)
    return total / max(count, 1)

# SAM3-related helper functions (specific to training_libero_msillm_sam.py)
def patch_modeagent_embed_visual_obs_for_msillm(model: LightningModule, compress_gripper: bool = True, compress_rgb: bool = True) -> Optional[torch.nn.Module]:
    """
    Patch `MoDEAgent.embed_visual_obs` at runtime to:
      normalized -> unnormalize to [0,1] -> encode(no_grad) -> decode(with grad) -> renormalize
    Also stores reconstruction cache for SAM3-guided loss.

    Args:
        model: LightningModule to patch
        compress_gripper: If True, apply reconstruction to gripper image as well. If False, only static image.
        compress_rgb: If True, apply reconstruction to rgb_static image. If False, skip compression for static image.

    Returns the decoder module if patch applied, else None.
    """
    msillm = getattr(model, "msillm_model", None)
    if msillm is None:
        return None

    encoder = getattr(msillm, "encoder", None)
    decoder = getattr(msillm, "decoder", None)
    if encoder is None or decoder is None:
        log_rank_0("msillm_model is present but missing `.encoder` or `.decoder`; skipping forward-time patch.")
        return None

    # Ensure encoder doesn't update internal stats; decoder may train.
    encoder.eval()
    decoder.train()
    
    # Ensure MS-ILLM's device_setting is "forward" to allow forward() calls
    if hasattr(msillm, "update_tensor_devices"):
        try:
            msillm.update_tensor_devices("forward")
        except Exception:
            # If update_tensor_devices fails, try to set device_setting directly
            if hasattr(msillm, "_device_setting"):
                msillm._device_setting = "forward"

    orig = getattr(model, "embed_visual_obs", None)
    if orig is None or not callable(orig):
        return None

    def _reconstruct_normed(x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x01: (B, T, C, H, W) in [0, 1] range (Normalize transform removed)
        mean, std = clip_mean_std(x01.device, x01.dtype)

        b, t, c, h, w = x01.shape
        x01_bt = x01.reshape(b * t, c, h, w)

        # MS-ILLM requires images to be divisible by 64 during training
        # Resize to nearest multiple of 64 if needed
        # 224 needs to be resized to 64's multiple (e.g., 224 -> 240 or 256)
        factor = 64  # MS-ILLM requires 64's multiple
        if h % factor != 0 or w % factor != 0:
            # Resize to nearest multiple of 16
            new_h = ((h + factor - 1) // factor) * factor
            new_w = ((w + factor - 1) // factor) * factor
            x01_bt_resized = F.interpolate(x01_bt, size=(new_h, new_w), mode='bilinear', align_corners=False)
            resize_needed = True
        else:
            x01_bt_resized = x01_bt
            resize_needed = False

        # CRITICAL: Follow MS-ILLM's forward process manually (from hific.py:491-524)
        # encoder -> hyper_analysis -> hyper_bottleneck -> hyper_synthesis -> latent_bottleneck -> decoder
        # All steps except decoder should be no_grad to ensure only decoder trains
        # IMPORTANT: Explicitly delete intermediate tensors to save memory (especially when compress_gripper=True)
        #with torch.no_grad():
        # Step 1: Encode image to latent (same as msillm.forward line 498)
        latent = encoder(x01_bt_resized)
        # Delete input after encoding to free memory
        del x01_bt_resized
        
        # Step 2: Hyperprior analysis (same as msillm.forward line 501)
        hyper_latent = msillm.hyper_analysis(latent)
        hyper_latent, _ = msillm.hyper_bottleneck(hyper_latent)
        
        # Step 3: Hyperprior synthesis (mean and scale) (same as msillm.forward lines 507-508)
        means = msillm.hyper_synthesis_mean(hyper_latent)
        scales = msillm.hyper_synthesis_scale(hyper_latent)
        
        # Delete hyper_latent to free memory (no longer needed after synthesis)
        del hyper_latent
        
        # Step 4: Latent bottleneck (quantization) (same as msillm.forward line 509-510)
        quantized_latents, _ = msillm.latent_bottleneck(latent, scales, means=means)
        
        # Delete intermediate tensors to free memory (only quantized_latent is needed)
        del latent, means, scales
        
        # Step 5: Use quantized latents (same as msillm.forward line 521 for eval mode)
        # Note: We use quantized_latents (not STE) since encoder/intermediate steps are frozen
        quantized_latent = quantized_latents
        del quantized_latents  # Free memory, only quantized_latent is needed
        
        # CRITICAL: Detach quantized_latent to break gradient flow from encoder (which is frozen)
        # but allow decoder gradients to flow. This prevents the "does not require grad" error.
        #quantized_latent = quantized_latent.detach()
        
        # Step 6: Decode (WITH gradients - this is what we're training) (same as msillm.forward line 524)
        recon_resized = decoder(quantized_latent)
        recon_resized = recon_resized.clamp(0.0, 1.0)
        
        # Resize back to original size if needed
        if resize_needed:
            recon = F.interpolate(recon_resized, size=(h, w), mode='bilinear', align_corners=False)
            del recon_resized  # Free memory after resizing
        else:
            recon = recon_resized

        recon = recon.reshape(b, t, c, h, w)
        out = (recon - mean) / std
        return out, recon

    def _patched(self, rgb_static, rgb_gripper, latent_goal):  # type: ignore
        # Apply MS-ILLM reconstruction so decoder gradients flow.
        # Also stash GT + recon (in [0,1]) for optional SAM3-guided recon loss.
        self._msillm_last_recon_cache = {}

        # Only reconstruct rgb_static if configured
        if compress_rgb:
            rgb_static_normed, rgb_static_recon = _reconstruct_normed(rgb_static)
            self._msillm_last_recon_cache["rgb_static_gt"] = rgb_static.detach()
            self._msillm_last_recon_cache["rgb_static_recon"] = rgb_static_recon
            rgb_static = rgb_static_normed
        else:
            # Normalize static image even when not compressing (inputs are in [0, 1] range)
            mean, std = clip_mean_std(rgb_static.device, rgb_static.dtype)
            rgb_static = (rgb_static - mean) / std
        
        # Only reconstruct gripper if configured
        if compress_gripper:
            rgb_gripper_normed, rgb_gripper_recon = _reconstruct_normed(rgb_gripper)
            self._msillm_last_recon_cache["rgb_gripper_gt"] = rgb_gripper.detach()
            self._msillm_last_recon_cache["rgb_gripper_recon"] = rgb_gripper_recon
            rgb_gripper = rgb_gripper_normed
        else:
            # Normalize gripper image even when not compressing (inputs are in [0, 1] range)
            mean, std = clip_mean_std(rgb_gripper.device, rgb_gripper.dtype)
            rgb_gripper = (rgb_gripper - mean) / std

        # Call original embed_visual_obs (bound method) with reconstructed static and (optionally) gripper images.
        return orig(rgb_static, rgb_gripper, latent_goal)

    # type: ignore[method-assign]
    model.embed_visual_obs = types.MethodType(_patched, model)
    return decoder

@hydra.main(config_path="../conf", config_name="config_libero_msillm_sam")
def train(cfg: DictConfig) -> None:
    try:
        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache before initialization
        clear_cuda_cache()
        
        # Initialize components
        log_rank_0(f"\n{'='*60}")
        log_rank_0(f"Initializing training for seed {cfg.seed}")
        log_rank_0(f"{'='*60}")
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        
        # Log dataset info
        log_rank_0(f"DataModule initialized: {type(datamodule).__name__}")

        # Check for resume checkpoint (env var > config > auto-detect)
        last_checkpoint = None
        resume_path = os.environ.get('RESUME_CHECKPOINT') or cfg.get("resume_from_checkpoint")
        if resume_path:
            resume_path = Path(resume_path)
            if resume_path.exists():
                last_checkpoint = resume_path
                log_rank_0(f"Resuming from checkpoint: {last_checkpoint}")
            else:
                log_rank_0(f"Checkpoint not found: {resume_path}, trying auto-detection")
        
        if last_checkpoint is None:
            last_checkpoint = get_last_checkpoint(Path.cwd())
            if last_checkpoint:
                log_rank_0(f"Auto-detected checkpoint: {last_checkpoint}")

        # Initialize model and load MS-ILLM
        model = hydra.utils.instantiate(cfg.model)
        msillm_model, _msillm_decoder = load_msillm_from_torchhub(cfg)
        if msillm_model is not None:
            setattr(model, "msillm_model", msillm_model)

        # Make custom loss cfg available to built-in hooks (e.g., sam3_weighted_recon_loss)
        global _CUSTOM_LOSS_CFG
        _CUSTOM_LOSS_CFG = cfg.custom_loss if "custom_loss" in cfg else None

        # Optional: patch training_step to include a user-defined custom loss term
        custom_loss_fn, custom_loss_weight, custom_loss_log_name = _load_custom_loss_hook(cfg)
        if custom_loss_fn is not None and custom_loss_weight != 0.0:
            _patch_training_step_with_custom_loss(
                model,
                loss_fn=custom_loss_fn,
                weight=custom_loss_weight,
                log_name=custom_loss_log_name,
            )
        
        # Load checkpoint if resuming
        if last_checkpoint:
            checkpoint = torch.load(last_checkpoint.as_posix(), map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if 'epoch' in checkpoint:
                log_rank_0(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                log_rank_0(f"Checkpoint global_step: {checkpoint['global_step']}")
            
            # Verify MS-ILLM was loaded from checkpoint
            if hasattr(model, "msillm_model") and model.msillm_model is not None:
                msillm_model = model.msillm_model
                _msillm_decoder = extract_compression_modules(msillm_model)[1]
                msillm_keys = [k for k in model.state_dict().keys() if k.startswith("msillm_model")]
                log_rank_0(f"MS-ILLM loaded from checkpoint: {len(msillm_keys)} params")

        # Verify MS-ILLM is attached
        if msillm_model is not None:
            if not hasattr(model, "msillm_model") or model.msillm_model is None:
                setattr(model, "msillm_model", msillm_model)
            msillm_keys = [k for k in model.state_dict().keys() if k.startswith("msillm_model")]
            log_rank_0(f"MS-ILLM attached: {len(msillm_keys)} params will be saved")
            
            # Log MS-ILLM model details
            encoder, decoder = extract_compression_modules(msillm_model)
            if encoder is not None:
                enc_params = count_params(encoder)
                log_rank_0(f"  MS-ILLM encoder: {enc_params[0]:,} total params ({enc_params[1]:,} trainable)")
            if decoder is not None:
                dec_params = count_params(decoder)
                log_rank_0(f"  MS-ILLM decoder: {dec_params[0]:,} total params ({dec_params[1]:,} trainable)")
        
        # Configure which modules to train/freeze based on config
        # Default: train only vision encoders and MS-ILLM decoder
        train_vision_encoders = cfg.get("train_vision_encoders", True)
        train_msillm_encoder = cfg.get("train_msillm_encoder", False)
        train_msillm_decoder = cfg.get("train_msillm_decoder", True)
        
        # Freeze/unfreeze vision encoders
        if train_vision_encoders:
            freeze_all_except_vision_encoders(model)
            log_rank_0("Both vision encoders (static_resnet and gripper_resnet) are trainable")
        else:
            set_requires_grad(model, False)
            log_rank_0("All model parameters are frozen")

        # Configure MS-ILLM encoder/decoder training
        # 1. train_msillm_encoder=True, train_msillm_decoder=False: decoder 제외 모두 학습 가능
        # 2. train_msillm_encoder=False, train_msillm_decoder=True: decoder만 학습 가능
        # 3. train_msillm_encoder=True, train_msillm_decoder=True: 모두 학습 가능
        # Note: MS-ILLM parameters are not in configure_optimizers by default, so we need to add them explicitly
        compression_decoder = None
        if msillm_model is not None:
            # Step 1: Freeze all MS-ILLM parameters first
            set_requires_grad(msillm_model, False)
            
            # Step 2: Unfreeze all if encoder training is enabled
            if train_msillm_encoder:
                set_requires_grad(msillm_model, True)
            
            # Step 3: Set decoder based on train_msillm_decoder
            encoder, decoder = extract_compression_modules(msillm_model)
            if decoder is not None:
                set_requires_grad(decoder, train_msillm_decoder)
                compression_decoder = decoder if train_msillm_decoder else None
            
            # Step 4: Add trainable MS-ILLM parameters to optimizer
            # If encoder training is enabled, add all MS-ILLM (decoder will be filtered if frozen)
            # If only decoder training, add decoder only
            if train_msillm_encoder:
                # Add entire MS-ILLM model (decoder will be filtered out if frozen by patch_optimizer_to_only_train_selected)
                patch_optimizer_to_only_train_selected(model, extra_trainable_module=msillm_model)
            elif compression_decoder is not None:
                # Only decoder is trainable, add decoder only
                patch_optimizer_to_only_train_selected(model, extra_trainable_module=compression_decoder)
            
            log_rank_0(f"MS-ILLM: encoder={train_msillm_encoder}, decoder={train_msillm_decoder}")

        # Patch embed_visual_obs to route images through MS-ILLM encoder(no_grad)/decoder(grad) in forward.
        compress_gripper = cfg.msillm.get("compress_gripper", True) if "msillm" in cfg else True
        compress_rgb = cfg.msillm.get("compress_rgb", True) if "msillm" in cfg else True
        patch_modeagent_embed_visual_obs_for_msillm(model, compress_gripper=compress_gripper, compress_rgb=compress_rgb)
        
        # Patch on_before_zero_grad to include MS-ILLM decoder grad norms in total_grad_norm calculation
        if msillm_model is not None and train_msillm_decoder:
            patch_on_before_zero_grad_for_msillm(model)
        
        # Load pretrained weights if configured
        if "pretrain_chk" in cfg:
            pretrain_chk = cfg.pretrain_chk
            # Check if it's a Hugging Face repo ID (contains "/" and doesn't look like a file path)
            if "/" in str(pretrain_chk) and not Path(pretrain_chk).exists():
                # Assume it's a Hugging Face repo ID (e.g., "mbreuss/MoDE_LIBERO_10")
                repo_id = str(pretrain_chk)
                filename = cfg.get("pretrain_chk_filename", "model_cleaned.safetensors")
                load_pretrained_weights_from_hf(model, repo_id, filename)
            else:
                # Use existing local file path loader
                initialize_pretrained_weights(model, cfg)
            
        # Setup training
        train_logger = setup_logger(cfg, model)
        
        # Determine MS-ILLM identifier (prefer checkpoint's identifier if resuming)
        msillm_info = get_msillm_identifier(cfg)
        if last_checkpoint:
            checkpoint_msillm_info = extract_msillm_identifier_from_checkpoint_path(last_checkpoint)
            if checkpoint_msillm_info:
                msillm_info = checkpoint_msillm_info
        
        callbacks = setup_callbacks(cfg.callbacks, msillm_info=msillm_info) + [LearningRateMonitor(logging_interval="step")]
        
        # Set unique working directory for each seed
        work_dir = Path.cwd() / f"seed_{cfg.seed}"
        work_dir.mkdir(exist_ok=True)
        os.chdir(work_dir)
        
        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "strategy": "ddp_find_unused_parameters_true",
            "accelerator": "gpu",
            "devices": cfg.trainer.devices,
            "use_distributed_sampler": True,
            "default_root_dir": work_dir,
            "sync_batchnorm": True,
            "log_every_n_steps": cfg.trainer.get("log_every_n_steps", 1),  # Log every step (default: 1 for step 0 logging)
        }
        
        # Log checkpoint save path
        checkpoint_callback = next((cb for cb in callbacks if hasattr(cb, 'dirpath')), None)
        if checkpoint_callback is not None:
            checkpoint_dir = Path(checkpoint_callback.dirpath).resolve() if checkpoint_callback.dirpath else work_dir / "saved_models"
            log_rank_0(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        # Log configuration
        log_rank_0(f"Training config for seed {cfg.seed}:\n{cfg}")
        log_rank_0(print_system_env_info())
        
        # Log training setup details
        log_rank_0(f"\n{'='*60}")
        log_rank_0(f"Training Setup Summary:")
        log_rank_0(f"{'='*60}")
        log_rank_0(f"Seed: {cfg.seed}")
        log_rank_0(f"Max epochs: {cfg.trainer.max_epochs}")
        log_rank_0(f"Devices: {cfg.trainer.devices}")
        log_rank_0(f"MS-ILLM identifier: {msillm_info if msillm_info else 'None'}")
        log_rank_0(f"Train vision encoders: {train_vision_encoders}")
        log_rank_0(f"Train MS-ILLM encoder: {train_msillm_encoder}")
        log_rank_0(f"Train MS-ILLM decoder: {train_msillm_decoder}")
        log_rank_0(f"Work directory: {work_dir}")
        log_rank_0(f"{'='*60}\n")
                
        # Clear CUDA cache again before training
        clear_cuda_cache()
        
        # Initialize trainer and train
        trainer = Trainer(**trainer_args)
        
        # Resume from checkpoint if available (for full resume including optimizer/scheduler state)
        fit_kwargs = {}
        if last_checkpoint is not None:
            fit_kwargs["ckpt_path"] = last_checkpoint.as_posix()
            log_rank_0(f"Resuming training from checkpoint: {last_checkpoint}")
            # Log checkpoint details
            checkpoint_info = torch.load(last_checkpoint.as_posix(), map_location='cpu', weights_only=False)
            if 'epoch' in checkpoint_info:
                log_rank_0(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
            if 'global_step' in checkpoint_info:
                log_rank_0(f"  Checkpoint global_step: {checkpoint_info['global_step']}")
            if 'lr_schedulers' in checkpoint_info:
                log_rank_0(f"  Checkpoint contains LR scheduler state")
        
        log_rank_0(f"\n{'='*60}")
        log_rank_0(f"Starting training...")
        log_rank_0(f"{'='*60}\n")
        
        trainer.fit(model, datamodule=datamodule, **fit_kwargs)
        
        log_rank_0(f"\n{'='*60}")
        log_rank_0(f"Training completed!")
        log_rank_0(f"{'='*60}\n")
    finally:
        # Clear CUDA cache one final time
        clear_cuda_cache()
        # Clean up
        cleanup_distributed()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    # Set environment variables (keep in __main__ to avoid import side-effects when this file
    # is imported as a module via `custom_loss.fn=...`).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Fix for PyTorch 2.6+ weights_only issue: force weights_only=False for checkpoint loading
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    try:
        train()
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

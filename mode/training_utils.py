"""
Training utility functions for MS-ILLM integration.
"""
import logging
import types
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import wandb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """Log message only on rank 0 process."""
    logger.info(*args, **kwargs)


def patch_on_before_zero_grad_for_msillm(model: LightningModule) -> None:
    """
    Patch `MoDEAgent.on_before_zero_grad` to include MS-ILLM decoder parameters in grad norm calculation.
    This ensures total_grad_norm includes gradients from MS-ILLM decoder when it's trainable.
    
    Args:
        model: LightningModule to patch
    """
    orig_on_before_zero_grad = getattr(model, "on_before_zero_grad", None)
    if orig_on_before_zero_grad is None or not callable(orig_on_before_zero_grad):
        return
    
    def _patched_on_before_zero_grad(optimizer=None):  # type: ignore
        """
        Extended gradient monitoring and logging for wrapped model with inner model, blocks, layers, and MS-ILLM decoder.
        """
        total_grad_norm = 0.0
        layer_grad_norms = {'input_layers': 0.0, 'blocks': {}}
        grad_stats = {'mean': [], 'median': [], 'max': [], 'min': []}
        
        # Compute grad norms for inner_model (original logic)
        for name, p in model.model.inner_model.named_parameters():
            if p.grad is not None:
                # Calculate total grad norm
                param_norm = p.grad.norm().item()
                total_grad_norm += param_norm ** 2
                
                # Log layer-wise grad norms
                if 'blocks' in name:
                    parts = name.split('.')
                    block_num = parts[1]
                    layer_name = '.'.join(parts[2:])  # Join the rest of the parts to get the layer name
                    
                    if block_num not in layer_grad_norms['blocks']:
                        layer_grad_norms['blocks'][block_num] = {}
                    
                    if layer_name not in layer_grad_norms['blocks'][block_num]:
                        layer_grad_norms['blocks'][block_num][layer_name] = 0.0
                    
                    layer_grad_norms['blocks'][block_num][layer_name] += param_norm ** 2
                else:
                    layer_grad_norms['input_layers'] += param_norm ** 2
                
                # Collect grad statistics
                grad_flat = p.grad.flatten()
                grad_stats['mean'].append(grad_flat.mean().item())
                grad_stats['median'].append(grad_flat.median().item())
                grad_stats['max'].append(grad_flat.max().item())
                grad_stats['min'].append(grad_flat.min().item())
        
        # Additionally compute grad norm for MS-ILLM decoder if present and trainable
        msillm_decoder_grad_norm_sq = 0.0
        msillm_model = getattr(model, "msillm_model", None)
        if msillm_model is not None:
            decoder = getattr(msillm_model, "decoder", None)
            if decoder is not None:
                for name, p in decoder.named_parameters():
                    if p.grad is not None and p.requires_grad:
                        param_norm = p.grad.norm().item()
                        msillm_decoder_grad_norm_sq += param_norm ** 2
        
        # Add MS-ILLM decoder grad norm to total
        total_grad_norm += msillm_decoder_grad_norm_sq
        
        # Calculate final norms and statistics
        total_grad_norm = total_grad_norm ** 0.5
        layer_grad_norms['input_layers'] = layer_grad_norms['input_layers'] ** 0.5
        msillm_decoder_grad_norm = msillm_decoder_grad_norm_sq ** 0.5 if msillm_decoder_grad_norm_sq > 0 else 0.0
        
        # Calculate norms for blocks and layers
        for block, layers in layer_grad_norms['blocks'].items():
            for layer, norm in layers.items():
                layer_grad_norms['blocks'][block][layer] = norm ** 0.5
        
        # Log total grad norm (now includes MS-ILLM decoder)
        model.log("debug/total_grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log input layers grad norm
        model.log("debug/input_layers_grad_norm", layer_grad_norms['input_layers'], on_step=True, on_epoch=False, sync_dist=True)
        
        # Log MS-ILLM decoder grad norm separately for debugging
        if msillm_decoder_grad_norm > 0:
            model.log("debug/msillm_decoder_grad_norm", msillm_decoder_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log block and layer-wise grad norms
        for block, layers in layer_grad_norms['blocks'].items():
            for layer, norm in layers.items():
                model.log(f"debug/block_{block}_{layer}_grad_norm", norm, on_step=True, on_epoch=False, sync_dist=True)
    
    # type: ignore[method-assign]
    model.on_before_zero_grad = types.MethodType(_patched_on_before_zero_grad, model)
    log_rank_0("Patched on_before_zero_grad to include MS-ILLM decoder grad norms")


def set_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool) -> None:
    """Set requires_grad for all parameters in a module."""
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def count_params(module: Optional[torch.nn.Module]) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    if module is None:
        return 0, 0
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def patch_optimizer_to_only_train_selected(
    model: LightningModule,
    *,
    extra_trainable_module: Optional[torch.nn.Module],
) -> None:
    """
    Make sure optimizers only see trainable params from the intended modules.
    This is implemented as a runtime patch to avoid editing the core model code.
    """
    if extra_trainable_module is None:
        return

    orig_configure_optimizers = getattr(model, "configure_optimizers", None)
    if orig_configure_optimizers is None or not callable(orig_configure_optimizers):
        return

    def _patched_configure_optimizers():  # type: ignore
        out = orig_configure_optimizers()
        # Lightning allows returning optimizer or dict with optimizer/scheduler.
        optimizer = out["optimizer"] if isinstance(out, dict) and "optimizer" in out else out
        # Filter out any frozen params from existing param groups.
        if hasattr(optimizer, "param_groups"):
            new_groups = []
            for g in optimizer.param_groups:
                params = [p for p in g.get("params", []) if getattr(p, "requires_grad", False)]
                if params:
                    g["params"] = params
                    new_groups.append(g)
            optimizer.param_groups = new_groups
        if hasattr(optimizer, "add_param_group"):
            params = [p for p in extra_trainable_module.parameters() if p.requires_grad]
            if params:
                # Get base LR from existing param groups
                base_lr = optimizer.param_groups[0].get("lr", 1e-5) if optimizer.param_groups else 1e-5
                optimizer.add_param_group({
                    "params": params, 
                    "weight_decay": 0.0,
                    "lr": base_lr
                })
                log_rank_0(f"Added MS-ILLM decoder to optimizer with LR={base_lr} (base LR={base_lr})")
        return out

    # type: ignore[method-assign]
    model.configure_optimizers = _patched_configure_optimizers


def freeze_all_except_vision_encoders(model: LightningModule) -> None:
    """
    Train both vision encoders for MoDEAgent:
      - `static_resnet` (trainable)
      - `gripper_resnet` (trainable)
    Everything else is frozen.
    """
    set_requires_grad(model, False)
    # Train both static_resnet and gripper_resnet
    static_resnet = getattr(model, "static_resnet", None)
    set_requires_grad(static_resnet, True)
    gripper_resnet = getattr(model, "gripper_resnet", None)
    set_requires_grad(gripper_resnet, True)


def clear_cuda_cache():
    """Clear CUDA cache and garbage collect unused memory."""
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


def get_msillm_identifier(cfg: DictConfig) -> str:
    """
    Build MS-ILLM identifier string for use in wandb names and checkpoint filenames.
    
    Returns:
        String like "msillm-NeuralCompression_v0.3.1-msillm_quality_1" or empty string if not configured.
    """
    if "msillm" not in cfg:
        return ""
    
    msillm_cfg = cfg.msillm
    hub_repo = msillm_cfg.get("hub_repo", "unknown")
    entrypoint = msillm_cfg.get("entrypoint", "unknown")
    # Extract repo name (e.g., "facebookresearch/NeuralCompression:v0.3.1" -> "NeuralCompression_v0.3.1")
    repo_name = hub_repo.split("/")[-1].replace(":", "_") if "/" in hub_repo else hub_repo
    # Sanitize for filename: replace special chars
    repo_name = repo_name.replace("/", "_").replace(":", "_")
    entrypoint = entrypoint.replace("/", "_").replace(":", "_")
    
    # Add compression status to identifier to distinguish save directories
    compress_gripper = msillm_cfg.get("compress_gripper", True)
    compress_rgb = msillm_cfg.get("compress_rgb", True)
    
    # Determine suffix based on compression settings
    if compress_gripper and not compress_rgb:
        # Only gripper compressed -> gripper_only
        compression_suffix = "_gripper_only"
    elif not compress_gripper and compress_rgb:
        # Only rgb_static compressed -> static_only
        compression_suffix = "_static_only"
    elif compress_gripper and compress_rgb:
        # Both compressed -> no suffix
        compression_suffix = ""
    else:
        # Neither compressed -> _none suffix to distinguish from both=True case
        compression_suffix = "_none"
    
    return f"msillm-{repo_name}-{entrypoint}{compression_suffix}"


def extract_msillm_identifier_from_checkpoint_path(checkpoint_path: Path) -> Optional[str]:
    """
    Extract MS-ILLM identifier from checkpoint filename.
    
    Examples:
        "msillm-NeuralCompression_v0.3.1-msillm_quality_1_epoch19.ckpt" -> "msillm-NeuralCompression_v0.3.1-msillm_quality_1"
        "msillm-NeuralCompression_main-msillm_quality_vlo1_epoch=epoch=21.ckpt" -> "msillm-NeuralCompression_main-msillm_quality_vlo1"
    
    Returns:
        MS-ILLM identifier string or None if not found.
    """
    filename = checkpoint_path.stem  # Get filename without extension
    # Pattern: msillm-<repo>-<entrypoint>_<rest>
    if filename.startswith("msillm-"):
        # Find the pattern: msillm-<repo>-<entrypoint>_<rest>
        # Split by underscore and find where epoch or other patterns start
        parts = filename.split("_")
        # Look for "epoch" or "epoch=" pattern to find where identifier ends
        identifier_parts = []
        for part in parts:
            if part.startswith("epoch"):
                break
            identifier_parts.append(part)
        
        if identifier_parts:
            # Reconstruct identifier (e.g., "msillm-NeuralCompression_v0.3.1-msillm_quality_1")
            identifier = "_".join(identifier_parts)
            # Handle case where entrypoint itself contains underscores (e.g., msillm_quality_vlo1)
            # The identifier should end before "epoch" or similar patterns
            return identifier
    
    return None


class WandbConfigCallback(Callback):
    """Callback to update wandb config after wandb is initialized."""
    def __init__(self, msillm_cfg: Optional[DictConfig] = None, msillm_info: str = ""):
        super().__init__()
        self.msillm_cfg = msillm_cfg
        self.msillm_info = msillm_info
    
    def on_train_start(self, trainer, pl_module):
        """Update wandb config when training starts (wandb.run is available)."""
        if wandb.run is None:
            log_rank_0("Warning: wandb.run is None, skipping wandb config update")
            return
        
        if self.msillm_cfg is not None and self.msillm_info:
            wandb.config.update({
                "msillm_hub_repo": self.msillm_cfg.get("hub_repo", "unknown"),
                "msillm_entrypoint": self.msillm_cfg.get("entrypoint", "unknown"),
                "msillm_pretrained": self.msillm_cfg.get("pretrained", False),
                "msillm_identifier": self.msillm_info,
            }, allow_val_change=True)
            log_rank_0(f"Updated wandb config with MS-ILLM information")


def setup_callbacks(callbacks_cfg: DictConfig, msillm_info: str = "") -> list[Callback]:
    """Setup callbacks with MS-ILLM identifier support."""
    result = []
    for cb_name, cb_cfg in callbacks_cfg.items():
        # Skip rollout_lh callback if it's disabled or causes import errors
        if cb_name == "rollout_lh":
            try:
                cb = hydra.utils.instantiate(cb_cfg)
                result.append(cb)
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"Skipping {cb_name} callback due to import error: {e}")
                continue
        else:
            # Update checkpoint filename in config before instantiation if MS-ILLM info is available
            if cb_name == "checkpoint" and msillm_info and "filename" in cb_cfg:
                original_filename = cb_cfg.get("filename", "epoch-{epoch:02d}")
                # Replace '=' with '-' in filename to avoid Hydra parsing issues
                original_filename = original_filename.replace("=", "-")
                # Prepend MS-ILLM info to filename
                cb_cfg["filename"] = f"{msillm_info}_{original_filename}"
            
            cb = hydra.utils.instantiate(cb_cfg)
            result.append(cb)
    return result


def setup_logger(cfg: DictConfig, model: LightningModule):
    """Setup logger with MS-ILLM identifier."""
    pathlib_cwd = Path.cwd()
    msillm_info = get_msillm_identifier(cfg)
    
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        seed = cfg.get("seed", None)
        if msillm_info:
            cfg.logger.name = msillm_info
            # Include seed in ID to make it unique and avoid conflicts
            # Only set ID if not already set by environment variable
            if cfg.logger.get("id") is None or cfg.logger.get("id") == "null":
                base_id = msillm_info.replace("/", "_").replace(":", "_")
                cfg.logger.id = f"{base_id}_seed{seed}" if seed is not None else base_id
        else:
            base_name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
            cfg.logger.name = base_name
            # Only set ID if not already set by environment variable
            if cfg.logger.get("id") is None or cfg.logger.get("id") == "null":
                base_id = cfg.logger.name.replace("/", "_").replace(":", "_")
                cfg.logger.id = f"{base_id}_seed{seed}" if seed is not None else base_id
                
    logger_instance = hydra.utils.instantiate(cfg.logger)
    
    return logger_instance


def extract_compression_modules(compression_model: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    Identify encoder/decoder components from an arbitrary compression model.
    """
    encoder = getattr(compression_model, "encoder", None)
    decoder = getattr(compression_model, "decoder", None)
    if encoder is None and hasattr(compression_model, "encode"):
        encoder = compression_model
    return encoder, decoder


def load_pretrained_weights_from_hf(model: LightningModule, repo_id: str, filename: str = "model_cleaned.safetensors") -> None:
    """
    Load pretrained weights from Hugging Face hub (safetensors format).
    
    Args:
        model: The model to load weights into.
        repo_id: Hugging Face repo ID (e.g., "mbreuss/MoDE_LIBERO_10").
        filename: Name of the safetensors file (default: "model_cleaned.safetensors").
    """
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    log_rank_0(f"Loading pretrained weights from Hugging Face: {repo_id}/{filename}")
    log_rank_0(f"Checkpoint path: {ckpt_path}")
    
    state_dict = load_file(ckpt_path)
    
    # Handle potential key prefixes (e.g., "state_dict.", "model.")
    # Note: Hugging Face checkpoints have 'model.' prefix removed during save (save_to_hf.py)
    # So 'inner_model.*' keys need to be mapped back to 'model.inner_model.*'
    fixed_state_dict = {}
    inner_model_keys_fixed = 0
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("state_dict."):
            k2 = k2[len("state_dict."):]
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        # Handle inner_model.* keys that need to be mapped to model.inner_model.*
        # (because save_to_hf.py removes 'model.' prefix, so inner_model.* -> model.inner_model.*)
        if k2.startswith("inner_model."):
            k2 = "model." + k2
            inner_model_keys_fixed += 1
        fixed_state_dict[k2] = v
    
    if inner_model_keys_fixed > 0:
        log_rank_0(f"Fixed {inner_model_keys_fixed} inner_model.* keys to model.inner_model.*")
    
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    log_rank_0(f"Loaded pretrained weights: {len(fixed_state_dict)} keys")
    if missing:
        log_rank_0(f"Missing keys (not loaded): {len(missing)} keys (first 10: {missing[:10]})")
    if unexpected:
        log_rank_0(f"Unexpected keys (ignored): {len(unexpected)} keys (first 10: {unexpected[:10]})")


def load_msillm_from_torchhub(cfg: DictConfig) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    Load a pretrained MS-ILLM model via torch.hub and return (model, decoder).

    Entry points are provided by `facebookresearch/NeuralCompression/hubconf.py`:
      - msillm_quality_1 ... msillm_quality_6
      - msillm_quality_vlo1, msillm_quality_vlo2
      - msillm_vqvae_xcit_p8_ch64_cb1024_h8

    Config (all optional):
      - msillm.hub_repo: e.g. "facebookresearch/NeuralCompression:v0.3.1" (default pinned)
      - msillm.entrypoint: e.g. "msillm_quality_1"
      - msillm.pretrained: bool (default True)
    """
    if "msillm" not in cfg:
        return None, None

    ms_cfg = cfg.msillm
    hub_repo = ms_cfg.hub_repo if "hub_repo" in ms_cfg else "facebookresearch/NeuralCompression:v0.3.1"
    entrypoint = ms_cfg.entrypoint if "entrypoint" in ms_cfg else "msillm_quality_1"
    pretrained = bool(ms_cfg.pretrained) if "pretrained" in ms_cfg else True

    try:
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=pretrained, verbose=False)
    except TypeError:
        # Some hub entries may not support `verbose`.
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=pretrained)

    _enc, dec = extract_compression_modules(msillm_model)
    if dec is None:
        log_rank_0(f"Loaded MS-ILLM via torch.hub ({hub_repo}, {entrypoint}) but could not find `.decoder`.")
    else:
        log_rank_0(f"Loaded MS-ILLM via torch.hub ({hub_repo}, {entrypoint}); decoder params: {count_params(dec)[0]}")
    return msillm_model, dec


def clip_mean_std(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Default normalization used by this repo's LIBERO transforms (CLIP mean/std).
    Shapes: (1, 1, 3, 1, 1) for broadcasting over (B, T, C, H, W).
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    return mean, std


def cleanup_distributed():
    """Cleanup distributed training resources"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

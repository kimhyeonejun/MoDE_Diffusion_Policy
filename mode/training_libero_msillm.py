import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os 
from typing import Optional, Tuple
import json
import time
import wandb
import hydra
from omegaconf import DictConfig
import torch
import types
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


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

import mode.models.mode_agent as models_m
from mode.utils.utils import get_last_checkpoint, initialize_pretrained_weights, print_system_env_info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# region agent log helpers
_DEBUG_LOG_PATH = "/home/khj20343/MoDE_Diffusion_Policy/.cursor/debug.log"
_DEBUG_SESSION_ID = "debug-session"

def _agent_log(*, runId: str, hypothesisId: str, location: str, message: str, data: dict) -> None:
    """Write one NDJSON line to the debug log (best-effort)."""
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never break training due to debug logging.
        pass
# endregion

def _set_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad

def _count_params(module: Optional[torch.nn.Module]) -> tuple[int, int]:
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

def _first_trainable_param_names(module: Optional[torch.nn.Module], max_items: int = 20) -> list[str]:
    if module is None:
        return []
    names: list[str] = []
    for n, p in module.named_parameters():
        if p.requires_grad:
            names.append(n)
            if len(names) >= max_items:
                break
    return names

def _patch_optimizer_to_only_train_selected(
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
                optimizer.add_param_group({"params": params, "weight_decay": 0.0})
        return out

    # type: ignore[method-assign]
    model.configure_optimizers = _patched_configure_optimizers

def _freeze_all_except_vision_encoders(model: LightningModule) -> None:
    """
    Train both vision encoders for MoDEAgent:
      - `static_resnet` (trainable)
      - `gripper_resnet` (trainable)
    Everything else is frozen.
    """
    _set_requires_grad(model, False)
    # Train both static_resnet and gripper_resnet
    static_resnet = getattr(model, "static_resnet", None)
    _set_requires_grad(static_resnet, True)
    gripper_resnet = getattr(model, "gripper_resnet", None)
    _set_requires_grad(gripper_resnet, True)

def _freeze_compression_encoder_only_train_decoder(compression_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """
    Freeze everything in compression model, then unfreeze decoder only.
    Returns the decoder module if found.
    """
    _set_requires_grad(compression_model, False)
    _enc, dec = extract_compression_modules(compression_model)
    _set_requires_grad(dec, True)
    return dec

class ImageCompressionTransform:
    """
    Lightweight transform that routes images through MS-ILLM compression model.
    For training, uses forward method to get output.image (same as training loop).
    Intended to mirror Phase 1 (image compression) so Phase 2 sees reconstructed images.
    """
    def __init__(
        self,
        compression_model: torch.nn.Module,
        device: str = "cpu",
    ):
        self.compression_model = compression_model.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x in shape (T, C, H, W) or (C, H, W).
        Returns a reconstructed tensor on CPU for downstream transforms.
        Uses forward method to get output.image (same as training).
        """
        original_shape = x.shape
        is_uint8 = x.dtype == torch.uint8
        x = x.to(self.device)
        if is_uint8:
            x = x.float() / 255.0

        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        x_flat = x.reshape(-1, *x.shape[-3:])  # Flatten to (B*T, C, H, W) if needed

        # Use forward method to get output.image (same as training loop)
        output = self.compression_model(x_flat)
        if hasattr(output, 'image'):
            recon = output.image.clamp(0.0, 1.0)
        elif hasattr(output, 'x_hat'):
            recon = output.x_hat.clamp(0.0, 1.0)
        else:
            # If output is just a tensor
            recon = output.clamp(0.0, 1.0) if isinstance(output, torch.Tensor) else x_flat

        # Reshape back to original batch structure
        if recon.shape[0] != batch_size:
            recon = recon.reshape(batch_size, *recon.shape[1:])

        # Best-effort reshape in case dimensions don't match
        if recon.shape != x.shape and recon.numel() == x.numel():
            recon = recon.view_as(x)

        if original_shape == recon.shape[1:]:
            recon = recon.squeeze(0)
        elif original_shape == recon.shape:
            pass  # Already correct shape
        else:
            # Try to match original shape
            recon = recon.reshape(original_shape) if recon.numel() == x.numel() else recon
        
        recon = recon.float().clamp(0, 1).cpu()
        return recon

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

@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)

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
    return f"msillm-{repo_name}-{entrypoint}"

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
    
    # Set tags for wandb before instantiation (to avoid triggering wandb.init early)
    # WandbLogger accepts tags parameter during initialization
    # if msillm_info and cfg.logger.get("_target_", "").endswith("WandbLogger"):
    #     existing_tags = cfg.logger.get("tags", [])
    #     if not isinstance(existing_tags, list):
    #         existing_tags = []
    #     cfg.logger.tags = existing_tags + [msillm_info, "msillm-training"]
    
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
    try:
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
    except Exception as e:
        log_rank_0(f"Failed to load pretrained weights from Hugging Face {repo_id}: {e}")
        raise

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
        log_rank_0(f"Loaded MS-ILLM via torch.hub ({hub_repo}, {entrypoint}); decoder params: {_count_params(dec)[0]}")
    return msillm_model, dec

def _clip_mean_std(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Default normalization used by this repo's LIBERO transforms (CLIP mean/std).
    Shapes: (1, 1, 3, 1, 1) for broadcasting over (B, T, C, H, W).
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    return mean, std

def patch_modeagent_embed_visual_obs_for_msillm(model: LightningModule) -> Optional[torch.nn.Module]:
    """
    Patch `MoDEAgent.embed_visual_obs` at runtime to:
      normalized -> unnormalize to [0,1] -> encode(no_grad) -> decode(with grad) -> renormalize

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

    orig = getattr(model, "embed_visual_obs", None)
    if orig is None or not callable(orig):
        return None

    did_log = {"v": False}

    def _reconstruct_normed(x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm: (B, T, C, H, W) normalized by CLIP mean/std
        mean, std = _clip_mean_std(x_norm.device, x_norm.dtype)
        x01 = (x_norm * std + mean).clamp(0.0, 1.0)

        b, t, c, h, w = x01.shape
        x01_bt = x01.reshape(b * t, c, h, w)

        with torch.no_grad():
            z = encoder(x01_bt)
        recon = decoder(z)
        if recon.shape != x01_bt.shape and recon.numel() == x01_bt.numel():
            recon = recon.view_as(x01_bt)
        recon = recon.clamp(0.0, 1.0)

        recon = recon.reshape(b, t, c, h, w)
        out = (recon - mean) / std
        return out

    def _patched(self, rgb_static, rgb_gripper, latent_goal):  # type: ignore
        # Apply MS-ILLM reconstruction to both static and gripper images so decoder gradients flow from both.
        rgb_static = _reconstruct_normed(rgb_static)
        rgb_gripper = _reconstruct_normed(rgb_gripper)

        if not did_log["v"]:
            did_log["v"] = True
            _agent_log(
                runId="joint-train",
                hypothesisId="msillm-forward",
                location="mode/training_libero_msillm.py:patch_modeagent_embed_visual_obs_for_msillm",
                message="Applied forward-time MS-ILLM recon to both static and gripper images (encoder no_grad, decoder grad)",
                data={
                    "rgb_static_shape": list(rgb_static.shape),
                    "rgb_gripper_shape": list(rgb_gripper.shape),
                    "gripper_compression": True,
                },
            )

        # Call original embed_visual_obs (bound method) with reconstructed static and gripper images.
        return orig(rgb_static, rgb_gripper, latent_goal)

    # type: ignore[method-assign]
    model.embed_visual_obs = types.MethodType(_patched, model)
    return decoder

def attach_compression_to_datamodule(datamodule, compression_model: Optional[torch.nn.Module], compression_cfg: Optional[DictConfig]) -> None:
    """
    Prepend the image compression transform to datamodule transforms so Phase 2
    consumes reconstructed images. Uses compression_model's compress/decompress or forward method.
    """
    if compression_model is None or not hasattr(datamodule, "transforms") or datamodule.transforms is None:
        return

    device = "cpu"
    if compression_cfg and "inference_device" in compression_cfg:
        device = compression_cfg.inference_device

    # If transforms are still configs, instantiate them so we can prepend callables.
        datamodule.transforms = hydra.utils.instantiate(datamodule.transforms)

    # Use the full compression model with forward method (same as training)
    transform = ImageCompressionTransform(compression_model=compression_model, device=device)
    for split in ("train", "val"):
        if split not in datamodule.transforms:
            continue
        # Apply compression transform to both static and gripper images
        if "rgb_static" in datamodule.transforms[split]:
            datamodule.transforms[split]["rgb_static"] = [transform] + list(datamodule.transforms[split]["rgb_static"])
        if "rgb_gripper" in datamodule.transforms[split]:
            datamodule.transforms[split]["rgb_gripper"] = [transform] + list(datamodule.transforms[split]["rgb_gripper"])
    
    log_rank_0("Injected image compression transform into datamodule transforms (both static and gripper images, using forward method).")


@hydra.main(config_path="../conf", config_name="config_libero_msillm")
def train(cfg: DictConfig) -> None:
    try:
        # Setup environment
        os.environ['HYDRA_FULL_ERROR'] = '1'
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Fix for PyTorch 2.6+ weights_only issue: force weights_only=False for checkpoint loading
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        
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
                enc_params = _count_params(encoder)
                log_rank_0(f"  MS-ILLM encoder: {enc_params[0]:,} total params ({enc_params[1]:,} trainable)")
            if decoder is not None:
                dec_params = _count_params(decoder)
                log_rank_0(f"  MS-ILLM decoder: {dec_params[0]:,} total params ({dec_params[1]:,} trainable)")
        
        # Configure which modules to train/freeze based on config
        # Default: train only vision encoders and MS-ILLM decoder
        train_vision_encoders = cfg.get("train_vision_encoders", True)
        train_msillm_encoder = cfg.get("train_msillm_encoder", False)
        train_msillm_decoder = cfg.get("train_msillm_decoder", True)
        
        # Freeze/unfreeze vision encoders
        if train_vision_encoders:
            _freeze_all_except_vision_encoders(model)
            log_rank_0("Both vision encoders (static_resnet and gripper_resnet) are trainable")
        else:
            _set_requires_grad(model, False)
            log_rank_0("All model parameters are frozen")

        # Configure MS-ILLM encoder/decoder training
        compression_decoder = None
        if msillm_model is not None:
            encoder, decoder = extract_compression_modules(msillm_model)
            
            # Freeze/unfreeze MS-ILLM encoder
            if encoder is not None:
                _set_requires_grad(encoder, train_msillm_encoder)
                log_rank_0(f"MS-ILLM encoder: {'trainable' if train_msillm_encoder else 'frozen'}")
            
            # Freeze/unfreeze MS-ILLM decoder
            if decoder is not None:
                _set_requires_grad(decoder, train_msillm_decoder)
                compression_decoder = decoder if train_msillm_decoder else None
                log_rank_0(f"MS-ILLM decoder: {'trainable' if train_msillm_decoder else 'frozen'}")
            
            # If decoder is trainable, add it to optimizer
            if compression_decoder is not None:
                _patch_optimizer_to_only_train_selected(model, extra_trainable_module=compression_decoder)

        # Patch embed_visual_obs to route images through MS-ILLM encoder(no_grad)/decoder(grad) in forward.
        patch_modeagent_embed_visual_obs_for_msillm(model)

        # region agent log
        _agent_log(
            runId="pre-fix",
            hypothesisId="B",
            location="mode/training_libero_msillm.py:train",
            message="Policy frozen except vision encoders (+ optional MS-ILLM decoder patch)",
            data={
                "policy_type": type(model).__name__,
                "policy_total_params": _count_params(model)[0],
                "policy_trainable_params": _count_params(model)[1],
                "policy_trainable_param_names_head": _first_trainable_param_names(model, max_items=20),
                "compression_decoder_present": compression_decoder is not None,
                "compression_decoder_trainable_params": _count_params(compression_decoder)[1] if compression_decoder is not None else 0,
            },
        )
        # endregion
        
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
        
        # Create wandb config callback to update config when training starts
        msillm_cfg_for_callback = cfg.msillm if msillm_info and "msillm" in cfg else None
        wandb_config_callback = WandbConfigCallback(msillm_cfg=msillm_cfg_for_callback, msillm_info=msillm_info) if msillm_info else None
        
        callbacks = setup_callbacks(cfg.callbacks, msillm_info=msillm_info) + [LearningRateMonitor(logging_interval="step")]
        if wandb_config_callback is not None:
            callbacks.append(wandb_config_callback)
        
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
                
    except Exception as e:
        logger.error(f"Training failed for seed {cfg.seed}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clear CUDA cache one final time
        clear_cuda_cache()
        # Clean up
        cleanup_distributed()
        if wandb.run is not None:
            wandb.finish()

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    # Add repo to path
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
    
    try:
        train()
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

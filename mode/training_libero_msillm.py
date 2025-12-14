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
from mode.utils.utils import get_git_commit_hash, get_last_checkpoint, initialize_pretrained_weights, print_system_env_info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# region agent log helpers
_DEBUG_LOG_PATH = "/home/hjkim/MoDE_Diffusion_Policy/.cursor/debug.log"
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
        try:
            if hasattr(optimizer, "param_groups"):
                new_groups = []
                for g in optimizer.param_groups:
                    params = [p for p in g.get("params", []) if getattr(p, "requires_grad", False)]
                    if params:
                        g["params"] = params
                        new_groups.append(g)
                optimizer.param_groups = new_groups
        except Exception:
            pass
        if hasattr(optimizer, "add_param_group"):
            params = [p for p in extra_trainable_module.parameters() if p.requires_grad]
            if params:
                optimizer.add_param_group({"params": params, "weight_decay": 0.0})
        return out

    # type: ignore[method-assign]
    model.configure_optimizers = _patched_configure_optimizers

def _freeze_all_except_vision_encoders(model: LightningModule) -> None:
    """
    Train ONLY the vision encoders for MoDEAgent:
      - `static_resnet`
      - `gripper_resnet`
    Everything else is frozen.
    """
    _set_requires_grad(model, False)
    for attr in ("static_resnet", "gripper_resnet"):
        m = getattr(model, attr, None)
        _set_requires_grad(m, True)

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
    Lightweight transform that routes images through a frozen encoder/decoder pair.
    Intended to mirror Phase 1 (image compression) so Phase 2 sees reconstructed images.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval() if decoder is not None else None
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x in shape (T, C, H, W) or (C, H, W).
        Returns a reconstructed tensor on CPU for downstream transforms.
        """
        original_shape = x.shape
        is_uint8 = x.dtype == torch.uint8
        x = x.to(self.device)
        if is_uint8:
            x = x.float() / 255.0

        if x.dim() == 3:
            x = x.unsqueeze(0)

        z = self.encoder(x)
        recon = self.decoder(z) if self.decoder is not None else z

        # Best-effort reshape in case encoder/decoder flattens the batch dimension
        if recon.shape != x.shape and recon.numel() == x.numel():
            recon = recon.view_as(x)

        if original_shape == recon.shape[1:]:
            recon = recon.squeeze(0)
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
                original_filename = cb_cfg.get("filename", "epoch={epoch:02d}")
                # Prepend MS-ILLM info to filename
                cb_cfg["filename"] = f"{msillm_info}_{original_filename}"
            
            cb = hydra.utils.instantiate(cb_cfg)
            result.append(cb)
    return result

def setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    
    # Build MS-ILLM identifier string
    msillm_info = get_msillm_identifier(cfg)
    
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        # Use MS-ILLM info as name if available, otherwise use default path-based name
        if msillm_info:
            cfg.logger.name = msillm_info
            cfg.logger.id = msillm_info.replace("/", "_").replace(":", "_")
        else:
            base_name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
            cfg.logger.name = base_name
            cfg.logger.id = cfg.logger.name.replace("/", "_").replace(":", "_")
    
    # Instantiate logger first (before setting tags to avoid struct mode issues)
    logger_instance = hydra.utils.instantiate(cfg.logger)
    
    # Set tags after instantiation (WandbLogger supports this)
    if msillm_info:
        msillm_tags = [msillm_info, "msillm-training"]
        # Try to set tags via logger instance
        if hasattr(logger_instance, 'tags'):
            existing_tags = logger_instance.tags if logger_instance.tags else []
            if isinstance(existing_tags, list):
                logger_instance.tags = existing_tags + msillm_tags
            else:
                logger_instance.tags = msillm_tags
        # Also try via experiment API
        if hasattr(logger_instance, 'experiment') and logger_instance.experiment is not None:
            try:
                current_tags = getattr(logger_instance.experiment, 'tags', []) or []
                if isinstance(current_tags, list):
                    logger_instance.experiment.tags = current_tags + msillm_tags
                else:
                    logger_instance.experiment.tags = msillm_tags
            except:
                pass  # Some wandb versions may not support this
    
    # Add MS-ILLM config to wandb config
    if hasattr(logger_instance, 'experiment') and logger_instance.experiment is not None:
        if "msillm" in cfg:
            msillm_cfg = cfg.msillm
            config_dict = {
                "msillm_hub_repo": msillm_cfg.get("hub_repo", "unknown"),
                "msillm_entrypoint": msillm_cfg.get("entrypoint", "unknown"),
                "msillm_pretrained": msillm_cfg.get("pretrained", False),
                "msillm_identifier": msillm_info,
            }
            try:
                # Try to access config as an object with update method
                config_obj = logger_instance.experiment.config
                if hasattr(config_obj, 'update'):
                    config_obj.update(config_dict)
                elif callable(config_obj):
                    # If config is a function, it might need to be called or accessed differently
                    # Try using wandb.config directly if available
                    if wandb.run is not None:
                        wandb.config.update(config_dict)
                else:
                    # Fallback: try to update via wandb.config
                    if wandb.run is not None:
                        wandb.config.update(config_dict)
            except Exception as e:
                # Fallback: use wandb.config directly if experiment.config doesn't work
                try:
                    if wandb.run is not None:
                        wandb.config.update(config_dict)
                except:
                    pass  # Some wandb versions may not support this
    
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
        fixed_state_dict = {}
        for k, v in state_dict.items():
            k2 = k
            if k2.startswith("state_dict."):
                k2 = k2[len("state_dict."):]
            if k2.startswith("model."):
                k2 = k2[len("model."):]
            fixed_state_dict[k2] = v
        
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
        # Apply MS-ILLM reconstruction before the vision encoders so decoder gradients flow from policy loss.
        rgb_static = _reconstruct_normed(rgb_static)
        rgb_gripper = _reconstruct_normed(rgb_gripper)

        if not did_log["v"]:
            did_log["v"] = True
            _agent_log(
                runId="joint-train",
                hypothesisId="msillm-forward",
                location="mode/training_libero_msillm.py:patch_modeagent_embed_visual_obs_for_msillm",
                message="Applied forward-time MS-ILLM recon (encoder no_grad, decoder grad)",
                data={
                    "rgb_static_shape": list(rgb_static.shape),
                    "rgb_gripper_shape": list(rgb_gripper.shape),
                },
            )

        # Call original embed_visual_obs (bound method) with reconstructed tensors.
        return orig(rgb_static, rgb_gripper, latent_goal)

    # type: ignore[method-assign]
    model.embed_visual_obs = types.MethodType(_patched, model)
    return decoder

def attach_compression_to_datamodule(datamodule, compression_model: Optional[torch.nn.Module], compression_cfg: Optional[DictConfig]) -> None:
    """
    Prepend the image compression transform to datamodule transforms so Phase 2
    consumes reconstructed images.
    """
    if compression_model is None or not hasattr(datamodule, "transforms") or datamodule.transforms is None:
        return

    encoder, decoder = extract_compression_modules(compression_model)
    if encoder is None:
        log_rank_0("Image compression model provided but no encoder found; skipping transform injection.")
        return

    device = "cpu"
    if compression_cfg and "inference_device" in compression_cfg:
        device = compression_cfg.inference_device

    # If transforms are still configs, instantiate them so we can prepend callables.
    try:
        datamodule.transforms = hydra.utils.instantiate(datamodule.transforms)
    except Exception as e:
        log_rank_0(f"Failed to instantiate transforms before injection: {e}")
        return

    transform = ImageCompressionTransform(encoder=encoder, decoder=decoder, device=device)
    for split in ("train", "val"):
        if split not in datamodule.transforms:
            continue
        for key in ("rgb_static", "rgb_gripper"):
            if key in datamodule.transforms[split]:
                datamodule.transforms[split][key] = [transform] + list(datamodule.transforms[split][key])
    log_rank_0("Injected image compression transform into datamodule transforms.")

def run_image_compression_phase(cfg: DictConfig, base_work_dir: Path) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """
    Optional Phase 1 training: train or load the image compression module.
    Expects an `image_compression` section in the Hydra config.
    """
    if "image_compression" not in cfg:
        return None, None

    comp_cfg = cfg.image_compression
    log_rank_0("Starting Phase 1: Image Compression Module")

    datamodule_cfg = comp_cfg.datamodule if "datamodule" in comp_cfg else cfg.datamodule
    trainer_cfg = comp_cfg.trainer if "trainer" in comp_cfg else cfg.trainer

    comp_datamodule = hydra.utils.instantiate(datamodule_cfg)
    comp_model = hydra.utils.instantiate(comp_cfg.model)

    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="A",
        location="mode/training_libero_msillm.py:run_image_compression_phase",
        message="Phase1 compression model instantiated",
        data={
            "comp_model_type": type(comp_model).__name__,
            "comp_model_total_params": _count_params(comp_model)[0],
            "comp_model_trainable_params": _count_params(comp_model)[1],
        },
    )
    # endregion

    # Train only decoder for the image compression model
    decoder = _freeze_compression_encoder_only_train_decoder(comp_model)

    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="A",
        location="mode/training_libero_msillm.py:run_image_compression_phase",
        message="Phase1 compression model frozen except decoder",
        data={
            "decoder_found": decoder is not None,
            "comp_model_trainable_params_after": _count_params(comp_model)[1],
            "decoder_trainable_params": _count_params(decoder)[1] if decoder is not None else 0,
            "decoder_trainable_param_names_head": _first_trainable_param_names(decoder, max_items=10),
        },
    )
    # endregion

    comp_logger = setup_logger(comp_cfg, comp_model) if "logger" in comp_cfg else setup_logger(cfg, comp_model)
    callbacks_cfg = comp_cfg.callbacks if "callbacks" in comp_cfg else cfg.callbacks
    comp_msillm_info = get_msillm_identifier(comp_cfg if "msillm" in comp_cfg else cfg)
    comp_callbacks = setup_callbacks(callbacks_cfg, msillm_info=comp_msillm_info) + [LearningRateMonitor(logging_interval="step")]

    work_dir = base_work_dir / "phase1_image_compression"
    work_dir.mkdir(exist_ok=True)

    trainer_args = {
        **{k: v for k, v in trainer_cfg.items()},
        "logger": comp_logger,
        "callbacks": comp_callbacks,
        "default_root_dir": work_dir,
        "accelerator": trainer_cfg.get("accelerator", "gpu"),
        "devices": trainer_cfg.devices if "devices" in trainer_cfg else cfg.trainer.devices,
        "strategy": "ddp_find_unused_parameters_true",
    }

    trainer = Trainer(**trainer_args)
    trainer.fit(comp_model, datamodule=comp_datamodule)

    ckpt_path = None
    if getattr(trainer, "checkpoint_callback", None) is not None:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        log_rank_0(f"Phase 1 best checkpoint: {ckpt_path}")

    clear_cuda_cache()
    return comp_model, ckpt_path

@hydra.main(config_path="../conf", config_name="config_libero_msillm")
def train(cfg: DictConfig) -> None:
    try:
        # Setup environment
        os.environ['HYDRA_FULL_ERROR'] = '1'
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache before initialization
        clear_cuda_cache()
        
        # Initialize components
        log_rank_0(f"\nInitializing training for seed {cfg.seed}")
        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # Check if we're resuming from checkpoint
        last_checkpoint = get_last_checkpoint(Path.cwd())
        if last_checkpoint is None:
            model = hydra.utils.instantiate(cfg.model)
            # Load MS-ILLM (pretrained) if configured and attach as a submodule so Lightning moves it with the model.
            msillm_model, _msillm_decoder = load_msillm_from_torchhub(cfg)
            if msillm_model is not None:
                setattr(model, "msillm_model", msillm_model)
        else:
            # Load model from checkpoint (this should include msillm_model if it was saved)
            model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(last_checkpoint.as_posix())
            # Check if msillm_model was loaded from checkpoint
            if hasattr(model, "msillm_model") and model.msillm_model is not None:
                log_rank_0("MS-ILLM model loaded from checkpoint")
                msillm_model = model.msillm_model
                _msillm_decoder = extract_compression_modules(msillm_model)[1]
            else:
                # MS-ILLM not in checkpoint, load from torch.hub (shouldn't happen if checkpoint was saved correctly)
                log_rank_0("MS-ILLM model not found in checkpoint, loading from torch.hub")
                msillm_model, _msillm_decoder = load_msillm_from_torchhub(cfg)
                if msillm_model is not None:
                    setattr(model, "msillm_model", msillm_model)

        # Verify MS-ILLM is attached to model (for checkpoint saving)
        if msillm_model is not None:
            if not hasattr(model, "msillm_model") or model.msillm_model is None:
                log_rank_0("WARNING: MS-ILLM model exists but is not attached to model. Attaching now...")
                setattr(model, "msillm_model", msillm_model)
            else:
                log_rank_0(f"MS-ILLM model is attached to model. Decoder params: {_count_params(_msillm_decoder)[0] if _msillm_decoder is not None else 0}")
        
        # Train only vision encoders (static/gripper resnets) for the policy
        _freeze_all_except_vision_encoders(model)

        # If MS-ILLM is present, train ONLY its decoder alongside the vision encoders.
        compression_decoder = None
        if msillm_model is not None:
            compression_decoder = _freeze_compression_encoder_only_train_decoder(msillm_model)
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
        msillm_info = get_msillm_identifier(cfg)
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
        }
        
        # Log checkpoint save path
        checkpoint_callback = next((cb for cb in callbacks if hasattr(cb, 'dirpath')), None)
        if checkpoint_callback is not None:
            checkpoint_dir = Path(checkpoint_callback.dirpath).resolve() if checkpoint_callback.dirpath else work_dir / "saved_models"
            log_rank_0(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        # Log configuration
        log_rank_0(f"Training config for seed {cfg.seed}:\n{cfg}")
        log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
        log_rank_0(print_system_env_info())
                
        # Clear CUDA cache again before training
        clear_cuda_cache()
        
        # Initialize trainer and train
        trainer = Trainer(**trainer_args)
        
        try:
            trainer.fit(model, datamodule=datamodule)
        except Exception as e:
            log_rank_0("\nDetailed Error Information:")
            log_rank_0("=" * 80)
            log_rank_0(f"Error Type: {type(e).__name__}")
            log_rank_0(f"Error Message: {str(e)}")
            log_rank_0("\nFull Traceback:")
            import traceback
            log_rank_0(''.join(traceback.format_tb(e.__traceback__)))
            log_rank_0("\nLocal Variables at Crash Point:")
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            log_rank_0(f"{traceback.extract_tb(tb)}")
            log_rank_0("=" * 80)
            raise e
                
    except Exception as e:
        logger.error(f"\nTraining failed for seed {cfg.seed}:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        raise e
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
        logger.error(f"\nTraining script failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        sys.exit(1)

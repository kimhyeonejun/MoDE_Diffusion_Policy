import contextlib
import logging
from pathlib import Path
from typing import Union
import importlib

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import pyhash
import torch
import types
from omegaconf import DictConfig
from safetensors.torch import load_file
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download

from mode.utils.utils import add_text, format_sftp_path

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def get_device(device_id):
    """
    Convert device_id to torch.device.
    
    Args:
        device_id: int (GPU index), 'cpu', or torch.device
        
    Returns:
        torch.device: The device object
    """
    if isinstance(device_id, torch.device):
        return device_id
    elif device_id == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(f"cuda:{device_id}")


def move_model_to_device(model, device):
    """
    Move model to specified device.
    
    Args:
        model: PyTorch model
        device: torch.device or device_id (int/'cpu')
        
    Returns:
        model: Model moved to device
    """
    device = get_device(device)
    if device.type == 'cuda':
        return model.cuda(device)
    else:
        return model.to(device)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_evaluation_checkpoint(cfg):
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    ).cuda()
    return pl_module


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pl_module_from_checkpoint(
    filepath: Union[Path, str],
    epoch: int = 1,
    overwrite_cfg: dict = {},
    use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
     # Load EMA weights if they exist and the flag is set
    if use_ema_weights:
        checkpoint_data = torch.load(ckpt_path)
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}

            model.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            print("Warning: No EMA weights found in checkpoint!")

    print(f"Finished loading model {ckpt_path}")
    return model



def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = get_device(device_id)

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")

    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    )
    # model = Hulc.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = move_model_to_device(model, device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    # Check if checkpoint is a Hugging Face repo ID (contains "/" and doesn't look like a file path)
    checkpoint_str = str(checkpoint)
    is_hf_repo = "/" in checkpoint_str and not Path(checkpoint_str).exists() and not Path(checkpoint_str).is_absolute()
    
    print(f"[get_default_mode_and_env] checkpoint: {checkpoint_str}")
    
    if is_hf_repo:
        # Hugging Face repo ID (e.g., "mbreuss/MoDE_LIBERO_10")
        repo_id = str(checkpoint)
        filename = "model_cleaned.safetensors"  # Default filename for Hugging Face checkpoints
        
        # Use default config instead of trying to load from Hugging Face
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize("../../conf")
        
        try:
            def_cfg = hydra.compose(config_name="config_libero")
        except:
            # Fallback: try to load from train_folder if it exists
            train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
            if train_cfg_path.exists():
                def_cfg = OmegaConf.load(train_cfg_path)
            else:
                raise FileNotFoundError(f"Could not find config. Tried config_libero.yaml and {train_cfg_path}")
        
        # Temporarily disable struct mode for lang_dataset if needed before merge
        if hasattr(def_cfg.datamodule.datasets, 'lang_dataset') and OmegaConf.is_struct(def_cfg.datamodule.datasets.lang_dataset):
            OmegaConf.set_struct(def_cfg.datamodule.datasets.lang_dataset, False)
            struct_was_enabled = True
        else:
            struct_was_enabled = False
        
        eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
        cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
        
        # Re-enable struct mode if it was enabled
        if struct_was_enabled and hasattr(cfg.datamodule.datasets, 'lang_dataset'):
            try:
                OmegaConf.set_struct(cfg.datamodule.datasets.lang_dataset, True)
            except:
                pass  # Don't fail if we can't re-enable struct mode
        
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize("../../conf/datamodule/datasets")
        
        device = get_device(device_id)
        cfg.datamodule.root_data_dir = dataset_path
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        
        if prep_dm_and_deps:
            data_module.prepare_data()
            data_module.setup()
            dataloader = data_module.val_dataloader()
            dataset = dataloader["lang"].dataset

            if lang_embeddings is None:
                # Get lang_folder only when needed, with fallback
                lang_folder = "lang_annotations"  # Default fallback
                try:
                    # Try to get from eval_cfg_overwrite first
                    if "datamodule" in eval_cfg_overwrite and "datasets" in eval_cfg_overwrite["datamodule"]:
                        if "lang_dataset" in eval_cfg_overwrite["datamodule"]["datasets"]:
                            if "lang_folder" in eval_cfg_overwrite["datamodule"]["datasets"]["lang_dataset"]:
                                lang_folder = eval_cfg_overwrite["datamodule"]["datasets"]["lang_dataset"]["lang_folder"]
                    # Otherwise try from config (but don't fail if struct mode blocks it)
                    elif hasattr(cfg.datamodule.datasets, 'lang_dataset'):
                        try:
                            lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
                        except (AttributeError, KeyError):
                            pass  # Use default
                except:
                    pass  # Use default
                
                lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

            if env is None:
                rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
                env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

        # Instantiate model (same as training)
        model_cfg = cfg.model
        model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
        if "ckpt_path" in model_cfg_dict:
            del model_cfg_dict["ckpt_path"]
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = hydra.utils.instantiate(model_cfg)
        
        # Load weights from Hugging Face (same as training_libero_msillm.py)
        print(f"[get_default_mode_and_env] Loading model weights from Hugging Face: {repo_id}/{filename}")
        try:
            print(f"[get_default_mode_and_env] Downloading from Hugging Face hub...")
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[get_default_mode_and_env] Downloaded to: {ckpt_path}")
            
            print(f"[get_default_mode_and_env] Loading safetensors file...")
            state_dict = load_file(ckpt_path)
            print(f"[get_default_mode_and_env] Loaded {len(state_dict)} keys from safetensors")
            
            # Handle potential key prefixes (same as training)
            # Note: Hugging Face checkpoints have 'model.' prefix removed during save (save_to_hf.py line 121)
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
                print(f"[get_default_mode_and_env] Fixed {inner_model_keys_fixed} inner_model.* keys to model.inner_model.*")
            
            print(f"[get_default_mode_and_env] Fixed state dict has {len(fixed_state_dict)} keys")
            print(f"[get_default_mode_and_env] Loading weights into model...")
            missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
            print(f"[get_default_mode_and_env] Loaded pretrained weights: {len(fixed_state_dict)} keys")
            if missing:
                print(f"[get_default_mode_and_env] Missing keys (not loaded): {len(missing)} keys (first 10: {missing[:10]})")
                if len(missing) > 50:
                    print(f"[get_default_mode_and_env] WARNING: Too many missing keys ({len(missing)}). Model may not be properly loaded!")
            if unexpected:
                print(f"[get_default_mode_and_env] Unexpected keys (ignored): {len(unexpected)} keys")
            
            print(f"[get_default_mode_and_env] Successfully loaded weights from Hugging Face!")
        except Exception as e:
            print(f"[get_default_mode_and_env] ERROR: Failed to load pretrained weights from Hugging Face {repo_id}: {e}")
            import traceback
            print(f"[get_default_mode_and_env] Traceback:\n{traceback.format_exc()}")
            raise
        
        model.freeze()
        model = move_model_to_device(model, device)
        model.eval()  # Ensure model is in eval mode
        print("[get_default_mode_and_env] Successfully loaded model from Hugging Face.")
        
        return model, env, data_module, lang_embeddings
    
    else:
        # Local checkpoint path
        train_cfg_path = Path(train_folder) / checkpoint / ".hydra/config.yaml"
        train_cfg_path = format_sftp_path(train_cfg_path)
        def_cfg = OmegaConf.load(train_cfg_path)
        eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
        cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
        lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize("../../conf/datamodule/datasets")
        # we don't want to use shm dataset for evaluation
        # GlobalHydra.instance().clear()
        # datasets_cfg = hydra.initialize("datamodule/datasets/vision_lang.yaml")
        # since we don't use the trainer during inference, manually set up data_module
        # cfg.datamodule.datasets = datasets_cfg
        device = get_device(device_id)
        cfg.datamodule.root_data_dir = dataset_path
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        if prep_dm_and_deps:
            data_module.prepare_data()
            data_module.setup()
            dataloader = data_module.val_dataloader()
            dataset = dataloader["lang"].dataset

            if lang_embeddings is None:
                lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

            if env is None:
                rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
                env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)


        # Load model from local checkpoint
        module_path = (Path(train_folder).expanduser())
        checkpoint_dir = module_path / checkpoint
        
        if not checkpoint_dir.is_dir():
            raise ValueError(f"not valid file path: {str(checkpoint_dir)}")
        
        config = get_config_from_dir(checkpoint_dir)
        
        # Look for weight files in the directory
        weight_file = None
        potential_files = [
            checkpoint_dir / "model_cleaned.safetensors",
            checkpoint_dir / "model.safetensors",
        ]
        # Also check for .ckpt files
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        if ckpt_files:
            potential_files.extend(ckpt_files)
        
        for p in potential_files:
            if p.exists():
                weight_file = p
                break
        
        if weight_file is None:
            raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir}. Looked for: {potential_files}")

        print(f"Loading model from {checkpoint_dir}")
        
        # Same approach as training_libero_msillm.py: instantiate model first, then load weights
        overwrite_cfg = eval_cfg_overwrite.model if "model" in eval_cfg_overwrite else {}
        load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
        # Don't set ckpt_path - we'll load weights manually (same as training)
        if "ckpt_path" in load_cfg:
            del load_cfg["ckpt_path"]
        model = hydra.utils.instantiate(load_cfg)
        
        # Load weights from file (same as training_libero_msillm.py line 718-719)
        print(f"Loading weights from {weight_file}")
        if weight_file.suffix == ".safetensors":
            state_dict = load_file(weight_file)
            # Handle potential key prefixes
            # Note: Hugging Face checkpoints have 'model.' prefix removed during save (save_to_hf.py line 121)
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
                print(f"Fixed {inner_model_keys_fixed} inner_model.* keys to model.inner_model.*")
            missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
            print(f"Loaded pretrained weights: {len(fixed_state_dict)} keys")
            if missing:
                print(f"Missing keys (not loaded): {len(missing)} keys")
            if unexpected:
                print(f"Unexpected keys (ignored): {len(unexpected)} keys")
        else:
            # .ckpt file - same as training_libero_msillm.py: load state_dict directly
            checkpoint = torch.load(weight_file.as_posix(), map_location='cpu', weights_only=False)
            missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Loaded weights from checkpoint: {len(checkpoint['state_dict'])} keys")
            if missing:
                print(f"Missing keys (not loaded): {len(missing)} keys")
            if unexpected:
                print(f"Unexpected keys (ignored): {len(unexpected)} keys")

        print(f"Finished loading model from {checkpoint_dir}")
        model.freeze()
        model = move_model_to_device(model, device)
        print("Successfully loaded model.")

        return model, env, data_module, lang_embeddings

# Helper functions for MS-ILLM
def extract_compression_modules(compression_model: torch.nn.Module):
    encoder = getattr(compression_model, "encoder", None)
    decoder = getattr(compression_model, "decoder", None)
    if encoder is None and hasattr(compression_model, "encode"):
        encoder = compression_model
    return encoder, decoder

def load_msillm_from_torchhub(cfg: DictConfig):
    if "msillm" not in cfg:
        return None, None

    ms_cfg = cfg.msillm
    hub_repo = ms_cfg.get("hub_repo", "facebookresearch/NeuralCompression:main")
    entrypoint = ms_cfg.get("entrypoint", "msillm_quality_1")
    pretrained = bool(ms_cfg.get("pretrained", True))

    try:
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=pretrained, verbose=False)
    except TypeError:
        msillm_model = torch.hub.load(hub_repo, entrypoint, pretrained=pretrained)

    _enc, dec = extract_compression_modules(msillm_model)
    return msillm_model, dec

def _clip_mean_std(device: torch.device, dtype: torch.dtype):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    return mean, std

def patch_modeagent_embed_visual_obs_for_msillm(model):
    msillm = getattr(model, "msillm_model", None)
    if msillm is None:
        return None

    # Check if compress and decompress methods exist
    if not hasattr(msillm, "compress") or not hasattr(msillm, "decompress"):
        # Fallback to encoder/decoder if compress/decompress don't exist
        encoder = getattr(msillm, "encoder", None)
        decoder = getattr(msillm, "decoder", None)
        if encoder is None or decoder is None:
            return None
        use_compress_decompress = False
    else:
        use_compress_decompress = True
        # If available, put model in compression mode (moves entropy bottlenecks to CPU)
        if hasattr(msillm, "update_tensor_devices"):
            try:
                msillm.update_tensor_devices("compress")
            except Exception as e:
                print(f"[WARNING] Failed to update tensor devices for compression: {e}")

    msillm.eval()

    orig = getattr(model, "embed_visual_obs", None)
    if orig is None or not callable(orig):
        return None

    def _reconstruct_normed(x_norm: torch.Tensor) -> torch.Tensor:
        mean, std = _clip_mean_std(x_norm.device, x_norm.dtype)
        x01 = (x_norm * std + mean).clamp(0.0, 1.0)

        b, t, c, h, w = x01.shape
        x01_bt = x01.reshape(b * t, c, h, w)

        with torch.no_grad():
            if use_compress_decompress:
                # Use compress/decompress (same as official MS-ILLM evaluation code)
                # Check if compress method has been wrapped (for BPP measurement)
                # This allows wrapper to capture latents even when called from patched function
                compress_method = getattr(msillm, "compress", None)
                if compress_method is not None:
                    compressed = compress_method(x01_bt, force_cpu=False)
                    recon = msillm.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
                else:
                    # Fallback if compress doesn't exist
                    recon = x01_bt
            else:
                recon = x01_bt
        
        if recon.shape != x01_bt.shape and recon.numel() == x01_bt.numel():
            recon = recon.view_as(x01_bt)

        recon = recon.reshape(b, t, c, h, w)
        
        # Store reconstructed image in [0, 1] range for video (recon is already in [0, 1])
        if hasattr(model, '_store_reconstructed_frame') and hasattr(model, 'msillm_model') and model.msillm_model is not None:
            # Take last timestep and first batch, convert to numpy [H, W, C] in [0, 255] range
            recon_frame = recon[0, -1] if recon.dim() == 5 else recon[0]  # [C, H, W]
            rgb_recon_np = (recon_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rgb_recon_np = np.rot90(rgb_recon_np, k=2, axes=(0, 1))  # Rotate 180 degrees
            model._last_reconstructed_frame = rgb_recon_np[..., ::-1]  # RGB to BGR for cv2
        
        out = (recon - mean) / std
        return out

    def _patched(self, rgb_static, rgb_gripper, latent_goal):
        rgb_static_recon = _reconstruct_normed(rgb_static)
        #rgb_gripper = _reconstruct_normed(rgb_gripper)
        return orig(rgb_static_recon, rgb_gripper, latent_goal)

    model.embed_visual_obs = types.MethodType(_patched, model)
    return msillm

def get_msillm_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    checkpoint_str = str(checkpoint)
    is_hf_repo = "/" in checkpoint_str and not Path(checkpoint_str).exists() and not Path(checkpoint_str).is_absolute()
    print(f"[get_msillm_mode_and_env] checkpoint: {checkpoint_str}")
    
    # Load config
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf")
    
    try:
        def_cfg = hydra.compose(config_name="config_libero_msillm")
    except:
        train_cfg_path = Path(train_folder).expanduser() / ".hydra/config.yaml"
        if train_cfg_path.exists():
            def_cfg = OmegaConf.load(train_cfg_path)
        else:
            raise FileNotFoundError(f"Could not find config. Tried config_libero_msillm.yaml and {train_cfg_path}")

    cfg = OmegaConf.merge(def_cfg, OmegaConf.create(eval_cfg_overwrite))
    
def get_msillm_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    checkpoint_str = str(checkpoint)
    is_hf_repo = "/" in checkpoint_str and not Path(checkpoint_str).exists() and not Path(checkpoint_str).is_absolute()
    print(f"[get_msillm_mode_and_env] checkpoint: {checkpoint_str}")
    
    # Load config
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf")
    
    try:
        def_cfg = hydra.compose(config_name="config_libero_msillm")
    except:
        train_cfg_path = Path(train_folder).expanduser() / ".hydra/config.yaml"
        if train_cfg_path.exists():
            def_cfg = OmegaConf.load(train_cfg_path)
        else:
            raise FileNotFoundError(f"Could not find config. Tried config_libero_msillm.yaml and {train_cfg_path}")

    cfg = OmegaConf.merge(def_cfg, OmegaConf.create(eval_cfg_overwrite))
    
    # Get lang_folder with fallback
    try:
        lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    except (AttributeError, KeyError):
        lang_folder = "lang_annotations"
    
    # Setup data module
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    
    device = get_device(device_id)
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    
    if prep_dm_and_deps:
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader["lang"].dataset
        
        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)
        
        if env is None:
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    # Get weight file
    if is_hf_repo:
        print(f"[get_msillm_mode_and_env] Downloading from Hugging Face: {checkpoint_str}")
        weight_file = Path(hf_hub_download(repo_id=checkpoint_str, filename="model_cleaned.safetensors"))
    else:
        ckpt_path = Path(train_folder).expanduser() / checkpoint
        print(f"Loading model from {ckpt_path}")
        
        # Find weight file
        potential_files = [ckpt_path] if ckpt_path.is_file() else [
            ckpt_path / "model.safetensors",
            ckpt_path / "model_cleaned.safetensors",
            *ckpt_path.glob("*.ckpt")
        ]
        weight_file = next((p for p in potential_files if p.exists()), None)
        if weight_file is None:
            raise FileNotFoundError(f"Could not find model weights in {ckpt_path}")

    # Instantiate model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("ckpt_path", None)
    model = hydra.utils.instantiate(OmegaConf.create(model_cfg))
    
    # Always load MS-ILLM model structure (needed for compress/decompress methods)
    # Weights will be loaded from checkpoint if available
    msillm_model, _ = load_msillm_from_torchhub(cfg)
    if msillm_model is not None:
        setattr(model, "msillm_model", msillm_model)
        print("Attached MS-ILLM model from torch hub (model structure)")
    
    # Load weights
    print(f"Loading weights from {weight_file}")
    if weight_file.suffix == ".safetensors":
        state_dict = load_file(weight_file)
        # Fix key mapping for Hugging Face checkpoints
        if is_hf_repo:
            fixed_state_dict = {}
            for k, v in state_dict.items():
                k2 = k.replace("state_dict.", "").replace("model.", "", 1)
                if k2.startswith("inner_model."):
                    k2 = "model." + k2
                fixed_state_dict[k2] = v
            state_dict = fixed_state_dict
        model.load_state_dict(state_dict, strict=False)
    else:
        sd = torch.load(weight_file, map_location='cpu', weights_only=False)
        # Try EMA weights first
        if "callbacks" in sd and "EMA" in sd["callbacks"] and "ema_weights" in sd["callbacks"]["EMA"]:
            ema_weights_list = sd["callbacks"]["EMA"]["ema_weights"]
            model_state_dict = model.state_dict()
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model_state_dict.items()) if i < len(ema_weights_list)}
            if ema_weights_dict:
                model.load_state_dict(ema_weights_dict, strict=False)
                print("Successfully loaded EMA weights from checkpoint!")
            else:
                model.load_state_dict(sd.get("state_dict", sd), strict=False)
                print("Loaded regular weights (EMA weights not found)")
        else:
            model.load_state_dict(sd.get("state_dict", sd), strict=False)
            print("Loaded regular weights")
    
    # Patch MS-ILLM and move to device
    patch_modeagent_embed_visual_obs_for_msillm(model)
    model.freeze()
    model = move_model_to_device(model, device)
    
    # Re-set compression mode after device move
    msillm_model = getattr(model, "msillm_model", None)
    if msillm_model is not None and hasattr(msillm_model, "update_tensor_devices"):
        try:
            msillm_model.update_tensor_devices("compress")
            print("Re-set MS-ILLM to compression mode after device move (partial-GPU)")
        except Exception as e:
            print(f"[WARNING] Failed to update tensor devices for compression after device move: {e}")
    
    print("Successfully loaded model.")
    return model, env, data_module, lang_embeddings, cfg

def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    s = sum(demo_task_counter.values())
    success_rate = (sum(live_task_counter.values()) / s if s > 0 else 0) * 100
    logger.info(f"Average Success Rate {mod} = {success_rate:.0f}%")
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs
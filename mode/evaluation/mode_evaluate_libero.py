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
from tqdm.auto import tqdm
import wandb
import torch
import torch.distributed as dist

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

from mode.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition, join_vis_lang, LangEmbeddings
from mode.evaluation.multistep_sequences import get_sequences
from mode.rollout.rollout_video import RolloutVideo
# Import LIBERO environment utilities
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

logger = logging.getLogger(__name__)
log_print = logging.getLogger(__name__).info  # Use logger.info as callable function

def get_action_stats(dm):
    action_mean = None
    action_std = None
    try:
        if hasattr(dm, 'train_datasets') and 'lang' in dm.train_datasets:
            concat_ds = dm.train_datasets['lang']
            print(f"[DEBUG] Inspecting Dataset for Action Stats...")
            if hasattr(concat_ds, 'datasets') and len(concat_ds.datasets) > 0:
                first_ds = concat_ds.datasets[0]
                if hasattr(first_ds, 'transforms'):
                    transforms = first_ds.transforms
                    print(f"[DEBUG] Transforms keys: {transforms.keys()}")
                    if 'actions' in transforms:
                        action_transforms = transforms['actions']
                        print(f"[DEBUG] Action Transforms: {action_transforms}")
                        
                        # Handle Compose or List
                        transforms_list = []
                        if isinstance(action_transforms, (list, tuple)):
                            transforms_list = action_transforms
                        elif hasattr(action_transforms, 'transforms'): # torchvision.transforms.Compose
                            transforms_list = action_transforms.transforms
                        else:
                            transforms_list = [action_transforms]
                            
                        for t in transforms_list:
                            print(f"[DEBUG] Checking transform: {type(t).__name__}")
                            if 'NormalizeVector' in type(t).__name__:
                                action_mean = t.mean.numpy()
                                action_std = t.std.numpy()
                                print(f"[DEBUG] FOUND NormalizeVector! mean={action_mean}, std={action_std}")
    except Exception as e:
        print(f"Error checking action stats: {e}")
        import traceback
        traceback.print_exc()
    return action_mean, action_std

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
        action_stats=(None, None),
    ):
        self.model = model
        self.transforms = transforms
        self.log_dir = log_dir
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
        self.device = device
        self.eval_sequences = None
        self.init_states_paths = []
        self.cfg = {}
        self.descriptions = []
        self.action_mean, self.action_std = action_stats
        
        # First, collect all descriptions
        for i in range(self.num_tasks):
            task_i = self.benchmark_instance.get_task(i)
            # Use problem_folder instead of task_names[i] to match LIBERO's get_task_init_states
            self.init_states_paths.append(
                os.path.join(self.init_states_folder, task_i.problem_folder, task_i.init_states_file)
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

        # print(f"number of rollouts: {len(successes)}")
        log_print(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
        if wandb.run is not None:
            wandb.log({"eval_lh/avg_seq_len": torch.tensor(result_array)})

        for success, task_name in zip(successes, self.task_names):
            log_print(f"eval_lh/sr_{task_name} with success {success}")
            if wandb.run is not None:
                wandb.log({f"eval_lh/sr_{task_name}": success})
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
        init_states = torch.load(init_states_path, weights_only=False)
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
            # Select one init state for this rollout
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
                
                # [DIAGNOSTIC] Check data statistics once per task
                if i == 0 and steps == 1:
                    print(f"\n[DIAGNOSTIC] Rollout {i} Step {steps} Data Stats:")
                    for k, v in data['rgb_obs'].items():
                        print(f"  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                    if 'lang' in goal and isinstance(goal['lang'], torch.Tensor):
                        v = goal['lang']
                        print(f"  goal['lang']: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                    if 'lang_text' in goal:
                        print(f"  goal['lang_text']: {goal['lang_text']}")

                # data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = model.step(data, goal)
                
                actions = actions.cpu().numpy()
                
                # Apply action denormalization if stats exist
                if self.action_mean is not None and self.action_std is not None:
                    actions = actions * self.action_std + self.action_mean

                # [DIAGNOSTIC] Check action statistics
                if i == 0 and steps == 1:
                    print(f"[DIAGNOSTIC] Model Output Actions (Post-Processing):")
                    print(f"  actions: shape={actions.shape}, min={actions.min():.4f}, max={actions.max():.4f}, mean={actions.mean():.4f}")
                    print(f"  First action: {actions[0] if len(actions.shape)>1 else actions}")
                    print("-" * 40)
                    
                obs, reward, done, info = env.step(actions)

                if store_video_this_rollout:
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


        success_rate = num_success / self.n_eval
        env.close()
        gc.collect()
        # print(f"[info] evaluate task {task_str} takes {t.get_elapsed_time():.1f} seconds")
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
        # Assuming data contains images in 'rgb_static' and 'rgb_gripper'
        # Manual preprocessing to ensure correctness
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        for key in data['rgb_obs']:
            # Get the image data
            x = data['rgb_obs'][key]
            
            # [CORRECTION] Do NOT rotate for model input.
            # The model was trained on raw (likely inverted) images from the dataset.
            # Rotation should ONLY be applied when saving videos for human viewing.
            # We removed the rotation logic here to ensure model receives consistent data.
            
            if len(x.shape) == 3:
                x = np.expand_dims(x, axis=0)
            
            # 1. Convert to tensor, float, permute
            x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(self.device)
            
            # 2. Scale to 0-1
            x = x / 255.0
            
            # 3. Resize if needed
            # rgb_static: 224x224 (env default)
            # rgb_gripper: 112x112 (from config)
            if key == 'rgb_gripper':
                x = torch.nn.functional.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            elif key == 'rgb_static':
                # Ensure 224x224 just in case
                if x.shape[-1] != 224 or x.shape[-2] != 224:
                    x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # 4. Normalize
            x = (x - mean) / std
            
            # Add batch dimension (B, T, C, H, W) -> here T=1
            data['rgb_obs'][key] = x.unsqueeze(0)

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
    
    # Ensure DataModule is setup to load statistics (action normalization, etc.)
    if not hasattr(dm, 'train_datasets') or not dm.train_datasets:
        dm.setup()

    # -------------------------------------------------------------------------
    # [DIAGNOSTIC] Verify if inner_model weights are correctly loaded
    # -------------------------------------------------------------------------
    print("\n" + "!"*60)
    print("[DIAGNOSTIC] Verifying inner_model weights loading...")
    
    try:
        # 1. Identify checkpoint file
        if "=" in cfg.checkpoint:
             sanitized_checkpoint = cfg.checkpoint.replace("=", "-")
             ckpt_path = Path(cfg.train_folder) / sanitized_checkpoint
             if not ckpt_path.exists():
                 ckpt_path = Path(cfg.train_folder) / cfg.checkpoint
        else:
             ckpt_path = Path(cfg.train_folder) / cfg.checkpoint
             
        # Handle Hugging Face cache path if needed (simplified check)
        if not ckpt_path.exists():
            # Try to find it in HF cache if it's a repo ID
            from huggingface_hub import try_to_load_from_cache
            if "/" in cfg.checkpoint: # Likely a repo ID
                cached_path = try_to_load_from_cache(cfg.checkpoint, "model_cleaned.safetensors")
                if cached_path:
                    ckpt_path = Path(cached_path)
        
        print(f"Inspecting checkpoint: {ckpt_path}")
        
        # 2. Load raw state_dict from file
        loaded_state_dict = None
        if str(ckpt_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            loaded_state_dict = load_file(ckpt_path)
        elif str(ckpt_path).endswith('.ckpt'):
            loaded_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        elif str(ckpt_path).endswith('.pt'):
            loaded_state_dict = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['state_dict']
        
        if loaded_state_dict:
            # 3. Find an inner_model key
            sample_key = None
            sample_val_file = None
            
            # Common patterns for inner_model keys
            patterns = ["model.inner_model.", "inner_model."]
            
            for k, v in loaded_state_dict.items():
                for p in patterns:
                    if p in k and "weight" in k and v.numel() > 100: # Check weights, not scalars
                        sample_key = k
                        sample_val_file = v
                        break
                if sample_key: break
            
            if sample_key:
                print(f"Found sample key in file: {sample_key}")
                
                # 4. Find corresponding parameter in loaded model
                # Map file key to model key
                model_param = None
                model_param_name = None
                
                # Heuristic mapping
                search_suffix = sample_key.split("inner_model.")[-1] # e.g. "blocks.0.attn.qkv.weight"
                
                for name, param in model.named_parameters():
                    if name.endswith(search_suffix):
                        model_param = param
                        model_param_name = name
                        break
                
                if model_param is not None:
                    print(f"Corresponding model parameter: {model_param_name}")
                    
                    # 5. Compare values
                    # Move file tensor to same device/dtype for comparison
                    val_file = sample_val_file.to(model_param.device).to(model_param.dtype)
                    val_model = model_param.data
                    
                    if torch.allclose(val_file, val_model, atol=1e-5):
                        print(f"✅ SUCCESS: Values MATCH! (First 5 params: {val_model.flatten()[:5]})")
                    else:
                        print(f"❌ FAILURE: Values MISMATCH!")
                        print(f"  File:  {val_file.flatten()[:5]}")
                        print(f"  Model: {val_model.flatten()[:5]}")
                        print(f"  Difference norm: {(val_file - val_model).norm().item()}")
                else:
                    print(f"❌ Could not find corresponding parameter in model for {sample_key}")
                    print("Available model keys (first 10):")
                    for i, (n, _) in enumerate(model.named_parameters()):
                        if i >= 10: break
                        print(f"  {n}")
            else:
                print("❌ Could not find any 'inner_model' keys in the checkpoint file.")
                print("Keys in checkpoint (first 10):", list(loaded_state_dict.keys())[:10])
        else:
            print("❌ Failed to load state_dict from checkpoint file.")

    except Exception as e:
        print(f"⚠️ Diagnostic check failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    print("!"*60 + "\n")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Check for Action Normalization
    # -------------------------------------------------------------------------
    action_mean, action_std = get_action_stats(dm)
    if action_mean is not None:
        print(f"[INFO] Found Action Normalization stats!")
        print(f"  Mean: {action_mean}")
        print(f"  Std: {action_std}")
    else:
        print("[INFO] No Action Normalization stats found. Assuming raw actions.")
    # -------------------------------------------------------------------------

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
        action_stats=(action_mean, action_std)
    )

    if cfg.log_wandb:
        os.makedirs(log_dir / "wandb", exist_ok=False)
        # Resolve interpolations before converting to dict for wandb config
        wandb_config = {
            "checkpoint": cfg.checkpoint,
            "benchmark_name": cfg.benchmark_name,
            "num_sequences": cfg.num_sequences,
            "n_eval": cfg.n_eval,
            "max_steps": cfg.max_steps,
            "task_embedding_format": cfg.task_embedding_format,
        }
        run = wandb.init(
            project='mode_libero_eval',
            entity=cfg.wandb_entity,
            config=wandb_config,
        )

    # Actually run the evaluation!
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    eval_libero.setup()
    eval_libero.start()

    if cfg.log_wandb:
        run.finish()
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)

if __name__ == "__main__":
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()

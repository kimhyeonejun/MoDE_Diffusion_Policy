import logging
import os
from typing import Any, Dict, Optional, Tuple, List, DefaultDict
from functools import partial
import seaborn as sns

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops 
import wandb

from torchmetrics import MeanMetric
from mode.models.edm_diffusion.gc_sampling import *
import mode.models.edm_diffusion.utils as utils
from mode.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from mode.callbacks.ema import EMA
from mode.models.perceptual_encoders.resnets import ResNetEncoderWithFiLM
from mode.models.perceptual_encoders.pretrained_resnets import FiLMResNet34Policy, FiLMResNet50Policy
from mode.models.networks.modedit import NoiseBlockMoE 
from mode.utils.lang_buffer import AdvancedLangEmbeddingBuffer


logger = logging.getLogger(__name__)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, submodule in model.named_modules():
        # Adjusting the condition to capture the desired layers
        if '.' not in name or name.count('.') <= 10:  # Can be adjusted based on your model structure
            # Counting parameters including submodules
            submodule_params = sum(p.numel() for p in submodule.parameters())
            if submodule_params > 0:
                print(f"{name} - Total Params: {submodule_params}")

    
class MoDEAgent(pl.LightningModule):
    """
    The lightning module used for training.
    """
    def __init__(
        self,
        language_goal: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        noise_scheduler: str = 'exponential',
        sigma_sample_density_type: str = 'loglogistic',
        use_perceiver: bool = False,
        obs_enc_dim: int = 512,
        cond_dim: int = 512,
        use_lr_scheduler: bool = True,
        ckpt_path=None,
        seed: int = 42,
        entropy_gamma: float = 0.0,
        router_z_delta: float = 0.001,
        start_from_pretrained: bool = False,
        use_text_not_embedding: bool = True,
        use_proprio: bool = False,
        act_window_size: int = 10,
        resnet_type: str = '18', 
    ):
        super(MoDEAgent, self).__init__()
        # Set obs_dim based on resnet_type
        obs_dim = 2048 if resnet_type == '50' else 512
        self.latent_dim = latent_dim
        model.inner_model.obs_dim = obs_dim
        self.model = hydra.utils.instantiate(model).to(self.device)

        # Select ResNet type based on parameter
        if resnet_type == '18':
            ResNetClass = ResNetEncoderWithFiLM  # You'll need to import this
        elif resnet_type == '34':
            ResNetClass = FiLMResNet34Policy
        elif resnet_type == '50':
            ResNetClass = FiLMResNet50Policy
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        self.static_resnet = ResNetClass(cond_dim)
        self.gripper_resnet = ResNetClass(cond_dim)
        self.use_perceiver = use_perceiver
        self.use_film_resnet = True
        self.use_text_not_embedding = use_text_not_embedding
        self.act_window_size = act_window_size
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        self.use_proprio = use_proprio
        # goal encoders
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        self.modality_scope = "lang"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.entropy_gamma = entropy_gamma
        self.router_z_delta = router_z_delta
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.start_from_pretrained = start_from_pretrained
        self.ema_callback_idx = None
        self.save_hyperparameters()

        # finetuning specific attributes
        self.expert_analysis_complete = False
        self.finetuning_config = None
        self.frozen_expert_mask = None

        if self.start_from_pretrained and ckpt_path is not None:
            self.load_pretrained_parameters(ckpt_path)

        self.need_precompute_experts_for_inference = False

        self.lang_buffer = AdvancedLangEmbeddingBuffer(self.language_goal, 10000)

    def load_pretrained_parameters(self, ckpt_path, strict: bool = False):
        """
        Load pretrained parameters specifically for ResNet50 with FiLM layers.
        Handles ImageNet pretrained weights and potential tensor format differences.
        """
        print(f"Loading parameters from {ckpt_path}")
        
        try:
            # Load state dict
            if os.path.isdir(ckpt_path):
                cleaned_safetensors = os.path.join(ckpt_path, "model_cleaned.safetensors")
                cleaned_pytorch = os.path.join(ckpt_path, "model_cleaned.pt")
                
                if os.path.exists(cleaned_safetensors):
                    from safetensors.torch import load_file
                    state_dict = load_file(cleaned_safetensors)
                    print("Loading from safetensors file")
                elif os.path.exists(cleaned_pytorch):
                    state_dict = torch.load(cleaned_pytorch)['state_dict']
                    print("Loading from PyTorch file")
                else:
                    raise FileNotFoundError(f"No cleaned weights found in {ckpt_path}")
            else:
                checkpoint_data = torch.load(ckpt_path)
                state_dict = checkpoint_data['state_dict']

            # Get current model state
            current_state = self.state_dict()
            new_state_dict = {}
            
            def expand_tensor(tensor, target_shape):
                """Expand tensor to target shape based on common patterns."""
                if len(tensor.shape) == 0:  # Empty tensor
                    return torch.zeros(target_shape, device=tensor.device)
                elif len(tensor.shape) == 1:
                    if len(target_shape) == 1:
                        # For batch norm weights/biases
                        if tensor.shape[0] != target_shape[0]:
                            # Handle case where tensor needs to be expanded
                            return tensor.repeat(target_shape[0] // tensor.shape[0])
                        return tensor
                    elif len(target_shape) == 4:
                        # For conv weights stored as 1D
                        return tensor.view(target_shape)
                elif len(tensor.shape) == 2 and len(target_shape) == 4:
                    # For conv weights stored as 2D
                    total_elements = np.prod(target_shape)
                    if tensor.numel() == total_elements:
                        return tensor.view(target_shape)
                return None

            def process_state_dict_entry(key, checkpoint_tensor, current_shape):
                """Process a single state dict entry, handling various tensor formats."""
                if checkpoint_tensor.shape == current_shape:
                    return checkpoint_tensor
                
                # Try to expand/reshape the tensor
                expanded = expand_tensor(checkpoint_tensor, current_shape)
                if expanded is not None:
                    print(f"Reshaped {key} from {checkpoint_tensor.shape} to {current_shape}")
                    return expanded
                    
                # For batch norm running stats
                if 'running_' in key and len(checkpoint_tensor.shape) != len(current_shape):
                    if checkpoint_tensor.numel() == np.prod(current_shape):
                        return checkpoint_tensor.view(current_shape)
                
                return None

            # Track different types of operations
            direct_copies = 0
            reshaped_tensors = 0
            skipped_tensors = 0
            
            # Process each key in the checkpoint
            for key, checkpoint_tensor in state_dict.items():
                # Skip CLIP visual stuff
                if 'visual' in key or 'clip' in key.lower():
                    continue
                    
                target_key = key
                # Check for key in current state dict
                if key not in current_state:
                    # Try common key mappings
                    for old_prefix, new_prefix in {
                        'img_encoder_image_wrist.': 'gripper_resnet.',
                        'img_encoder_image_secondary.': 'static_resnet.',
                        'img_encoder_image_primary.': 'static_resnet.',
                        'net.': 'gripper_resnet.resnet.',
                    }.items():
                        if key.startswith(old_prefix):
                            target_key = key.replace(old_prefix, new_prefix)
                            break
                
                if target_key in current_state:
                    current_shape = current_state[target_key].shape
                    processed_tensor = process_state_dict_entry(target_key, checkpoint_tensor, current_shape)
                    
                    if processed_tensor is not None:
                        new_state_dict[target_key] = processed_tensor
                        if processed_tensor.shape != checkpoint_tensor.shape:
                            reshaped_tensors += 1
                        else:
                            direct_copies += 1
                    else:
                        print(f"Skipping incompatible tensor {target_key}:")
                        print(f"  Checkpoint shape: {checkpoint_tensor.shape}")
                        print(f"  Target shape: {current_shape}")
                        skipped_tensors += 1
            
            # Print summary
            print(f"\nLoading Summary:")
            print(f"Direct copies: {direct_copies}")
            print(f"Reshaped tensors: {reshaped_tensors}")
            print(f"Skipped tensors: {skipped_tensors}")
            
            # Load the state dict
            missing, unexpected = self.load_state_dict(new_state_dict, strict=strict)
            
            if missing and not strict:
                print(f"\nMissing keys: {len(missing)}")
                print("Examples of missing keys:")
                for key in sorted(missing)[:10]:
                    print(f"  - {key}")
                if len(missing) > 10:
                    print(f"  ... and {len(missing)-10} more")
            
            print("\nSuccessfully loaded weights!")
            self.prepare_model_for_finetuning()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {ckpt_path}: {str(e)}")

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''

        optim_groups = self.get_optim_groups()

        #optim_groups = [
        #    {"params": self.model.inner_model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        # ]
        optim_groups.extend([
            {"params": self.static_resnet.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.gripper_resnet.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])
        if self.use_perceiver:
            optim_groups.extend([
                {"params": self.perceiver.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            ])
        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate, betas=self.optimizer_config.betas)
        # Optionally initialize the scheduler 
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer


    def on_before_zero_grad(self, optimizer=None):
        """
        Extended gradient monitoring and logging for wrapped model with inner model, blocks, and layers
        """
        total_grad_norm = 0.0
        layer_grad_norms = {'input_layers': 0.0, 'blocks': {}}
        grad_stats = {'mean': [], 'median': [], 'max': [], 'min': []}
        
        for name, p in self.model.inner_model.named_parameters():
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
        
        # Calculate final norms and statistics
        total_grad_norm = total_grad_norm ** 0.5
        layer_grad_norms['input_layers'] = layer_grad_norms['input_layers'] ** 0.5
        
        # Calculate norms for blocks and layers
        for block, layers in layer_grad_norms['blocks'].items():
            for layer, norm in layers.items():
                layer_grad_norms['blocks'][block][layer] = norm ** 0.5
        
        # Log total grad norm
        self.log("debug/total_grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log input layers grad norm
        self.log("debug/input_layers_grad_norm", layer_grad_norms['input_layers'], on_step=True, on_epoch=False, sync_dist=True)
        
        # Log block and layer-wise grad norms
        for block, layers in layer_grad_norms['blocks'].items():
            for layer, norm in layers.items():
                self.log(f"debug/block_{block}_{layer}_grad_norm", norm, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log grad statistics
        # for stat, values in grad_stats.items():
        #    self.log(f"debug/grad_{stat}", np.mean(values), on_step=True, on_epoch=False, sync_dist=True)

    def get_optim_groups(self):
        # Helper function to check if a parameter should use weight decay
        def use_weight_decay(name):
            return all(x not in name for x in ['bias', 'LayerNorm', 'embedding'])

        # Split parameters into two groups
        decay = []
        no_decay = []
        
        for name, param in self.model.inner_model.named_parameters():
            if use_weight_decay(name):
                decay.append(param)
            else:
                no_decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        return optim_groups

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Compute and return the training loss for the mode Agent.
        The training loss consists of the score matching loss of the diffusion model 
        and the contrastive loss of the CLIP model for the multimodal encoder.
        
        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch.
            
        Returns:
            loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0
        batch_sizes = []
        for self.modality_scope, dataset_batch in batch.items():
            # Compute the required embeddings
            perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

            act_loss = self.diffusion_loss(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
            )
            
            if self.entropy_gamma > 0:
                entropy_loss = self.model.inner_model.load_balancing_loss() 
                total_loss += entropy_loss * self.entropy_gamma

            if self.router_z_delta > 0:
                router_z_loss = self.model.inner_model.compute_router_z_loss()
                total_loss += self.router_z_delta * router_z_loss

            action_loss += act_loss
            total_loss += act_loss
            
            batch_sizes.append(dataset_batch["actions"].shape[0])
            total_bs += dataset_batch["actions"].shape[0]

        # Average losses across datasets
        batch_len = len(batch)
        total_loss = total_loss / batch_len
        action_loss = action_loss / batch_len
        total_bs 
        # Log metrics with the current batch size
        current_batch_size = sum(batch_sizes)
        # Log the metrics
        self._log_training_metrics(action_loss, total_loss,total_bs)
        if self.entropy_gamma > 0:
            self.log("train/load_balancing_loss", entropy_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        if self.router_z_delta > 0:
            self.log("train/router_z_delta", router_z_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> None:  # Change return type to None
        output = {}
        dataset_batch = batch
        perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)
        
        action_pred = self.denoise_actions(
            torch.zeros_like(latent_goal).to(self.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        
        actions = dataset_batch["actions"].to(self.device)
        pred_loss = torch.nn.functional.mse_loss(action_pred, actions)


        self._log_validation_metrics(pred_loss)

        output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        output["validation_loss"] = pred_loss
        self.log_expert_usage(self.model, self.current_epoch)
        return output
    
    def log_expert_usage(self, model, epoch):
        log_dir = self.logger.save_dir
        expert_usages = {}

        for name, module in self.model.inner_model.named_modules():
            if isinstance(module, NoiseBlockMoE):
                if module.total_tokens_processed > 0:
                    # Use get_expert_usage() instead of property
                    normalized_usage = module.get_expert_usage().cpu().numpy() / module.total_tokens_processed
                    expert_usages[name] = normalized_usage
                    module.reset_expert_usage()

        if expert_usages:
            # print(f"Logging expert usage for epoch {epoch}")
            # Convert list to numpy array
            expert_usage_data = np.array(list(expert_usages.values()))
            
            # Normalize each row independently
            row_sums = expert_usage_data.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-8)  # Avoid division by zero
            expert_usage_data_normalized = expert_usage_data / row_sums
            print(expert_usage_data_normalized)
            # Plotting the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                expert_usage_data_normalized, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                xticklabels=range(expert_usage_data_normalized.shape[1]), 
                yticklabels=[f'blocks.{i}' for i in range(expert_usage_data_normalized.shape[0])]
            )
            plt.xlabel('Expert Index')
            plt.ylabel('Block Number')
            plt.title(f'Expert Usage across Blocks (Epoch {epoch})')
            
            # Log the plot to wandb
            # Log to wandb with additional metadata
            self.logger.experiment.log({
                "MoE_utils/expert_usage_heatmap": wandb.Image(plt),
                "epoch": epoch
            })

            plt.close()
        else:
            print(f"No expert usage data to log for epoch {epoch}")

    def log_with_device_check(self, name, value, **kwargs):
        if torch.is_tensor(value) and value.device.type != "cuda":
            value = value.to(self.device)
        self.log(name, value, **kwargs)
    
    def _log_validation_metrics(self, pred_loss):
        """
        Log the validation metrics.
        """
        self.log(f"val_act/{self.modality_scope}_act_loss_pp", pred_loss, sync_dist=True)

      
    def compute_input_embeddings(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        latent_goal = None
        # last images are the randomly sampled future goal images for models learned with image goals 
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'] # [:, :-1]
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'] #[:, :-1]

        if self.use_text_not_embedding:
            # latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(dataset_batch["lang_text"]).to(rgb_static.dtype)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper, latent_goal)

        if self.use_proprio:
            perceptual_emb['robot_obs'] = dataset_batch['robot_obs']
        
        return perceptual_emb, latent_goal
    
    def embed_visual_obs(self, rgb_static, rgb_gripper, latent_goal):
        # reshape rgb_static and rgb_gripper
        rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
        rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')

        if self.use_film_resnet:
            static_tokens = self.static_resnet(rgb_static, latent_goal)
            gripper_tokens = self.gripper_resnet(rgb_gripper, latent_goal)
        else:
            static_tokens = self.static_resnet(rgb_static)
            gripper_tokens = self.gripper_resnet(rgb_gripper)

        # 4. compute the perceptual embeddings
        # first reshape the tokens
        static_tokens = einops.rearrange(static_tokens, '(b t) d -> b t d', b=rgb_static.shape[0])
        gripper_tokens = einops.rearrange(gripper_tokens, '(b t) d -> b t d', b=rgb_gripper.shape[0])
        token_seq = torch.cat([static_tokens, gripper_tokens], dim=1)
        perceptual_emb = {'state_images': token_seq}

        return perceptual_emb
 
    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True,batch_size=total_bs)
        
    
    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.latent_goal = None
        self.rollout_step_counter = 0
    
    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if self.use_text_not_embedding:
            # latent_goal = self.language_goal(goal["lang_text"])
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(obs["rgb_obs"]['rgb_static'].device)
        if self.need_precompute_experts_for_inference:
            self.precompute_expert_for_inference(latent_goal)
            self.need_precompute_experts_for_inference = False
        

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper, latent_goal)
        
        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions. 
        We only compute the sequence once every self.multistep steps to save computation and increase efficiency.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self(obs, goal)

            self.pred_action_seq = pred_action_seq  
            
        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action
    
    def precompute_expert_for_inference(self, goal=None):
        logger.info("Precomputing experts with sampling steps %d", self.num_sampling_steps)
        sigmas = self.get_noise_schedule(self.num_sampling_steps, self.noise_scheduler)[:-1]
        # iterate over the sigmas and precompute the experts
        for sigma in sigmas:
            self.model.inner_model.precompute_experts_for_inference(sigma, goal)

    def on_train_start(self)-> None:
        
        self.model.to(dtype=self.dtype)
        self.static_resnet.to(dtype=self.dtype)
        self.gripper_resnet.to(dtype=self.dtype)
        # self.perceiver.to(dtype=self.dtype)
        # self.language_goal.to(dtype=self.dtype)
        
        for idx, callback in enumerate(self.trainer.callbacks):
            if isinstance(callback, EMA):
                self.ema_callback_idx = idx
                break

    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss
    
    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")


    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")
        
    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")
        # self.model.inner_model.reset_expert_caches() if hasattr(self.model.inner_model, 'reset_expert_caches') else None
        # self.need_precompute_experts_for_inference = True

    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps*1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')
    
    def denoise_actions(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
        extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions 
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape): 
            latent_goal = latent_goal.unsqueeze(1) # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        if len(latent_goal.shape) == 2:
            goal = einops.rearrange(goal, 'b d -> 1 b d')

        x = torch.randn((len(latent_goal), self.act_window_size, 7), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions
    
    def prepare_model_for_finetuning(self):
        """Prepare model for efficient finetuning"""
        # Freeze router and unused experts
        self.model.inner_model.freeze_router()
        # pass

    def reset_expert_cache(self):
        self.model.inner_model.reset_all_caches() if hasattr(self.model.inner_model, 'reset_all_caches') else None

    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        latent_plan: torch.Tensor,
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'debugging':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # x_0 = sample_euler_visualization(self.model, state, x_t, goal, sigmas, self.scaler, self.working_dir, disable=True, extra_args={'keep_last_actions': True})
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7, self.device) # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def on_train_start(self)-> None:
        
        self.model.to(dtype=self.dtype)
        self.static_resnet.to(dtype=self.dtype)
        self.gripper_resnet.to(dtype=self.dtype)
        # self.perceiver.to(dtype=self.dtype)
        self.language_goal.to(dtype=torch.float32)

@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)

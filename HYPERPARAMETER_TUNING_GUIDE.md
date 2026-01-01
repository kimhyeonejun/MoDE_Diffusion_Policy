# Hyperparameter Tuning Guide for MS-ILLM Training

현재 training setting에서 변경 가능한 hyperparameter들을 정리한 문서입니다.

## 1. Optimizer Parameters (최적화 관련)

### 1.1 Learning Rate
- **Config path**: `model.optimizer.learning_rate`
- **Current default**: `1e-4`
- **Usage**: `LEARNING_RATE=5e-5 ./train_all_msillm_qualities.sh`
- **Hydra override**: `model.optimizer.learning_rate=5e-5`
- **Typical range**: `1e-5` to `5e-4`

### 1.2 Weight Decay
- **Config path**: `model.optimizer.transformer_weight_decay`, `model.optimizer.obs_encoder_weight_decay`
- **Current default**: `0.05` (both)
- **Hydra override**: 
  - `model.optimizer.transformer_weight_decay=0.1`
  - `model.optimizer.obs_encoder_weight_decay=0.1`
- **Typical range**: `0.0` to `0.1`

### 1.3 Optimizer Betas (AdamW)
- **Config path**: `model.optimizer.betas`
- **Current default**: `[0.9, 0.95]`
- **Hydra override**: `model.optimizer.betas=[0.9,0.99]`
- **Typical values**: `[0.9, 0.95]`, `[0.9, 0.99]`, `[0.9, 0.999]`

## 2. Learning Rate Scheduler Parameters

### 2.1 LR Scheduler Peak/Initial LR
- **Config path**: `model.lr_scheduler.lr_scheduler.init_lr`
- **Current default**: `1e-4`
- **Hydra override**: `model.lr_scheduler.lr_scheduler.init_lr=5e-5`
- **Note**: Usually matches `model.optimizer.learning_rate`

### 2.2 Initial LR Scale
- **Config path**: `model.lr_scheduler.lr_scheduler.init_lr_scale`
- **Current default**: `0.1`
- **Hydra override**: `model.lr_scheduler.lr_scheduler.init_lr_scale=0.05`
- **Description**: Ratio of initial learning rate to peak learning rate (warmup)

### 2.3 Final LR Scale
- **Config path**: `model.lr_scheduler.lr_scheduler.final_lr_scale`
- **Current default**: `1e-6`
- **Hydra override**: `model.lr_scheduler.lr_scheduler.final_lr_scale=1e-7`
- **Description**: Ratio of final learning rate to peak learning rate

### 2.4 Total Steps
- **Config path**: `model.lr_scheduler.lr_scheduler.total_steps`
- **Current default**: `45000`
- **Hydra override**: `model.lr_scheduler.lr_scheduler.total_steps=54000`
- **Description**: Total training steps for LR scheduler

### 2.5 Phase Ratio (Tri-stage scheduler)
- **Config path**: `model.lr_scheduler.lr_scheduler.phase_ratio`
- **Current default**: `"(0.02, 0.08, 0.9)"`
- **Hydra override**: `model.lr_scheduler.lr_scheduler.phase_ratio="(0.05, 0.1, 0.85)"`
- **Description**: Ratio of (warmup, constant, decay) phases

### 2.6 Use LR Scheduler
- **Config path**: `model.use_lr_scheduler`
- **Current default**: `True`
- **Hydra override**: `model.use_lr_scheduler=false`

## 3. Training Configuration

### 3.1 Batch Size
- **Config path**: `batch_size`
- **Current default**: `256`
- **Hydra override**: `batch_size=128`
- **Note**: Memory constrained? Try smaller values like `128`, `64`

### 3.2 Max Epochs
- **Config path**: `max_epochs`
- **Current default**: `100`
- **Hydra override**: `max_epochs=150`
- **Typical range**: `50` to `200`

### 3.3 Number of Workers (DataLoader)
- **Config path**: `num_workers`
- **Current default**: `0`
- **Hydra override**: `num_workers=4`
- **Note**: Increase for faster data loading (requires more CPU)

### 3.4 Seed
- **Config path**: `seed`
- **Current default**: `242`
- **Hydra override**: `seed=123`

### 3.5 Gradient Clipping
- **Config path**: `trainer.gradient_clip_val`
- **Current default**: Commented out (disabled)
- **Hydra override**: `trainer.gradient_clip_val=1.0`
- **Typical values**: `0.5`, `1.0`, `5.0`

### 3.6 Precision
- **Config path**: `trainer.precision`
- **Current default**: `bf16`
- **Hydra override**: `trainer.precision=16` or `trainer.precision=32`
- **Options**: `bf16`, `16`, `32`

### 3.7 Limit Training Batches (for debugging)
- **Config path**: `trainer.limit_train_batches`
- **Current default**: Not set (uses full dataset)
- **Hydra override**: `trainer.limit_train_batches=1000`
- **Note**: Useful for quick testing

### 3.8 Limit Validation Batches
- **Config path**: `trainer.limit_val_batches`
- **Current default**: `100`
- **Hydra override**: `trainer.limit_val_batches=50`

## 4. Model Architecture / Diffusion Parameters

### 4.1 Entropy Gamma (MoDE router entropy regularization)
- **Config path**: `model.entropy_gamma`
- **Current default**: `0.0` (for finetuning)
- **Hydra override**: `model.entropy_gamma=0.01`
- **Note**: Use `0.01` for training from scratch, `0.0` for finetuning

### 4.2 Router Z Delta
- **Config path**: `model.router_z_delta`
- **Current default**: `0.00`
- **Hydra override**: `model.router_z_delta=0.001`

### 4.3 Number of Sampling Steps (inference)
- **Config path**: `model.num_sampling_steps`
- **Current default**: `10`
- **Hydra override**: `model.num_sampling_steps=20`
- **Note**: Affects inference time, higher = better quality but slower

### 4.4 Sampler Type
- **Config path**: `model.sampler_type`
- **Current default**: `'ddim'`
- **Hydra override**: `model.sampler_type=euler`
- **Options**: `'ddim'`, `'euler'`, `'ancestral'`, `'euler_ancestral'`, `'dpmpp_2m'`, `'dpmpp_2m_sde'`

### 4.5 Sigma Min (noise schedule)
- **Config path**: `model.sigma_min`
- **Current default**: `0.001`
- **Hydra override**: `model.sigma_min=0.01`
- **Typical range**: `0.001` to `1.0`

### 4.6 Sigma Max (noise schedule)
- **Config path**: `model.sigma_max`
- **Current default**: `80`
- **Hydra override**: `model.sigma_max=100`
- **Typical range**: `50` to `100`

### 4.7 Noise Scheduler
- **Config path**: `model.noise_scheduler`
- **Current default**: `'exponential'`
- **Hydra override**: `model.noise_scheduler=linear`
- **Options**: `'exponential'`, `'linear'`

### 4.8 Sigma Sample Density Type
- **Config path**: `model.sigma_sample_density_type`
- **Current default**: `'loglogistic'`
- **Hydra override**: `model.sigma_sample_density_type=loguniform`
- **Options**: `'loglogistic'`, `'loguniform'`

### 4.9 Sigma Data
- **Config path**: `model.sigma_data`
- **Current default**: `0.5`
- **Hydra override**: `model.sigma_data=0.3`

## 5. MS-ILLM Specific Parameters

### 5.1 Compress Gripper
- **Config path**: `msillm.compress_gripper`
- **Current default**: `true`
- **Hydra override**: `msillm.compress_gripper=false`
- **Description**: Whether to compress gripper image or only static image

### 5.2 Train Vision Encoders
- **Config path**: `train_vision_encoders`
- **Current default**: `false`
- **Hydra override**: `train_vision_encoders=true`
- **Description**: Train static_resnet and gripper_resnet

### 5.3 Train MS-ILLM Encoder
- **Config path**: `train_msillm_encoder`
- **Current default**: `false`
- **Hydra override**: `train_msillm_encoder=true`
- **Warning**: Usually keep frozen

### 5.4 Train MS-ILLM Decoder
- **Config path**: `train_msillm_decoder`
- **Current default**: `true`
- **Hydra override**: `train_msillm_decoder=false`

## 6. Usage Examples

### Example 1: Change Learning Rate and Batch Size
```bash
LEARNING_RATE=5e-5 ./train_all_msillm_qualities.sh
# Then manually add batch_size override in the command, or modify script
```

### Example 2: Multiple Hyperparameter Overrides (Direct Hydra)
```bash
python mode/training_libero_msillm.py \
  msillm.entrypoint=msillm_quality_2 \
  model.optimizer.learning_rate=5e-5 \
  batch_size=128 \
  max_epochs=150 \
  model.entropy_gamma=0.01 \
  trainer.gradient_clip_val=1.0
```

### Example 3: Change LR Scheduler Settings
```bash
python mode/training_libero_msillm.py \
  msillm.entrypoint=msillm_quality_2 \
  model.lr_scheduler.lr_scheduler.init_lr=5e-5 \
  model.lr_scheduler.lr_scheduler.total_steps=54000 \
  model.lr_scheduler.lr_scheduler.phase_ratio="(0.05, 0.1, 0.85)"
```

### Example 4: Change Diffusion Parameters
```bash
python mode/training_libero_msillm.py \
  msillm.entrypoint=msillm_quality_2 \
  model.num_sampling_steps=20 \
  model.sigma_min=0.01 \
  model.sampler_type=euler
```

## 7. Recommended Hyperparameter Tuning Strategy

### Priority 1 (Most Impact)
1. **Learning Rate** (`model.optimizer.learning_rate`): Start with `1e-4`, try `5e-5`, `2e-4`
2. **Batch Size** (`batch_size`): Balance between memory and stability
3. **Max Epochs** (`max_epochs`): Ensure sufficient training

### Priority 2 (Moderate Impact)
4. **Weight Decay** (`model.optimizer.transformer_weight_decay`)
5. **LR Scheduler Settings** (`model.lr_scheduler.lr_scheduler.*`)
6. **Gradient Clipping** (`trainer.gradient_clip_val`)

### Priority 3 (Fine-tuning)
7. **Diffusion Parameters** (`model.sigma_min`, `model.sigma_max`, `model.num_sampling_steps`)
8. **Entropy Gamma** (`model.entropy_gamma`) - mainly for training from scratch
9. **Training Flags** (`train_vision_encoders`, `train_msillm_decoder`)

## 8. Notes

- All hyperparameters can be overridden using Hydra's command-line syntax
- Use `LEARNING_RATE` environment variable for learning rate (already supported in script)
- For other hyperparameters, you can either:
  1. Modify the config files directly
  2. Use Hydra command-line overrides
  3. Extend `train_all_msillm_qualities.sh` to support more environment variables
- Check `conf/config_libero_msillm.yaml` and `conf/model/mode_agent.yaml` for current defaults


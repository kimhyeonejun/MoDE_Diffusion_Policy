#!/bin/bash
# 평가 실행 스크립트 예제

# 체크포인트 파일 경로를 환경 변수로 설정
export CHECKPOINT_PATH="msillm-NeuralCompression_main-msillm_quality_vlo2_gripper_only_epoch=34.ckpt"

# 또는 절대 경로 사용 (더 안전함)
# export CHECKPOINT_PATH="/home/hjkim/MoDE_Diffusion_Policy/saved_models/msillm-NeuralCompression_main-msillm_quality_1_epoch=00.ckpt"

# 평가 실행
cd /home/hjkim/MoDE_Diffusion_Policy
CUDA_VISIBLE_DEVICES=1 taskset -c 0-1 python mode/evaluation/mode_evaluate_libero_msillm.py \
    train_folder=/home/hjkim/MoDE_Diffusion_Policy/saved_models \
    dataset_path=/tmp \
    device=0 \
    benchmark_name=libero_10 \
    num_sequences=50 \
    max_steps=520 \
    n_eval=50 \
    num_videos=1 \
    log_wandb=false \
    use_reconstructed_video=true
#!/bin/bash

# Script to evaluate all MS-ILLM quality variants
# Based on: https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm
#
# This script evaluates ALL MS-ILLM quality variants:
# - msillm_quality_vlo1, vlo2, 1, 2, 3, 4, 5, 6
# All use pretrain_chk from config_libero_msillm.yaml
#
# Usage:
#   ./evaluate_all_msillm_qualities.sh
#   DEVICE=1 ./evaluate_all_msillm_qualities.sh  # Use GPU 1
#   LOG_WANDB=false ./evaluate_all_msillm_qualities.sh  # Disable wandb

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/mode/evaluation/mode_evaluate_libero_msillm.py"
DEVICE=${DEVICE:-0}
LOG_WANDB=${LOG_WANDB:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MS-ILLM Quality Evaluation Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Based on: https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm${NC}"
echo ""
echo -e "Device: ${DEVICE}"
echo -e "Wandb logging: ${LOG_WANDB}"
echo ""

# Function to run evaluation
run_evaluation() {
    local checkpoint=$1
    local msillm_entrypoint=$2
    local description=$3
    local index=$4
    local total=$5
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[${index}/${total}] ${description}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "Checkpoint: ${checkpoint}"
    echo -e "MS-ILLM Entrypoint: ${msillm_entrypoint}"
    echo ""
    
    # Build command
    local cmd="python ${EVAL_SCRIPT} checkpoint=${checkpoint} device=${DEVICE} log_wandb=${LOG_WANDB}"
    
    # Add MS-ILLM entrypoint override
    cmd="${cmd} eval_cfg_overwrite.msillm.entrypoint=${msillm_entrypoint}"
    # Ensure hub_repo is set to main branch for vlo variants
    if [[ "$msillm_entrypoint" == *"vlo"* ]]; then
        cmd="${cmd} eval_cfg_overwrite.msillm.hub_repo=facebookresearch/NeuralCompression:main"
    fi
    
    echo -e "${BLUE}Running: taskset -c 19,20 ${cmd}${NC}"
    echo ""
    
    # Run evaluation with CPU affinity
    local start_time=$(date +%s)
    if taskset -c 19,20 bash -c "${cmd}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✓ Successfully completed: ${description} (took ${duration}s)${NC}"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${RED}✗ Failed: ${description} (after ${duration}s)${NC}"
        return 1
    fi
    
    echo ""
    sleep 2  # Brief pause between evaluations
}

# Track results
TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=()

# Define all evaluations to run
EVALUATIONS=()

# Add ALL MS-ILLM quality variants (vlo1, vlo2, 1-6)
EVALUATIONS+=("null|msillm_quality_vlo1|MS-ILLM Quality VLO1")
EVALUATIONS+=("null|msillm_quality_vlo2|MS-ILLM Quality VLO2")
EVALUATIONS+=("null|msillm_quality_1|MS-ILLM Quality 1")
EVALUATIONS+=("null|msillm_quality_2|MS-ILLM Quality 2")
EVALUATIONS+=("null|msillm_quality_3|MS-ILLM Quality 3")
EVALUATIONS+=("null|msillm_quality_4|MS-ILLM Quality 4")
EVALUATIONS+=("null|msillm_quality_5|MS-ILLM Quality 5")
EVALUATIONS+=("null|msillm_quality_6|MS-ILLM Quality 6")

TOTAL=${#EVALUATIONS[@]}

# Run all evaluations
for i in "${!EVALUATIONS[@]}"; do
    IFS='|' read -r checkpoint entrypoint description <<< "${EVALUATIONS[$i]}"
    index=$((i + 1))
    
    if run_evaluation "${checkpoint}" "${entrypoint}" "${description}" ${index} ${TOTAL}; then
        ((PASSED++))
    else
        ((FAILED++))
        FAILED_TESTS+=("${description}")
    fi
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total: ${TOTAL}"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED}${NC}"
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- ${test}${NC}"
    done
else
    echo -e "${GREEN}Failed: 0${NC}"
fi
echo -e "${GREEN}========================================${NC}"

# Exit with error if any test failed
if [ ${FAILED} -gt 0 ]; then
    exit 1
fi


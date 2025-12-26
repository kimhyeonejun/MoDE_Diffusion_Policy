#!/bin/bash

# Submit SLURM jobs to train with all MS-ILLM quality variants.
# Each quality is submitted as a separate sbatch job.
# Usage:
#   ./train_all_msillm_qualities.sh
#   LOG_WANDB=false ./train_all_msillm_qualities.sh   # Disable wandb

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/mode/training_libero_msillm.py"
LOG_WANDB=${LOG_WANDB:-true}
CONDA_ENV=${CONDA_ENV:-mode_env}
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MS-ILLM Quality Training Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Based on: https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm${NC}"
echo ""
echo -e "Wandb logging: ${LOG_WANDB}"
echo ""

# Submit training as an sbatch job
submit_training() {
    set +e
    local msillm_entrypoint=$1
    local description=$2
    local index=$3
    local total=$4

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[${index}/${total}] Submitting: ${description}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "MS-ILLM Entrypoint: ${msillm_entrypoint}"
    echo ""

    # Sanitize job name
    local job_suffix="${msillm_entrypoint}"
    job_suffix=${job_suffix//[^A-Za-z0-9_.-]/-}
    local job_name="msillm-train-${job_suffix}"

    # Build command - use python from conda env directly
    local python_cmd="\${HOME}/anaconda3/envs/mode_env/bin/python"
    local cmd="${python_cmd} ${TRAIN_SCRIPT}"
    cmd="${cmd} msillm.entrypoint=${msillm_entrypoint}"
    if [[ "$msillm_entrypoint" == *"vlo"* ]]; then
        cmd="${cmd} msillm.hub_repo=facebookresearch/NeuralCompression:main"
    fi
    if [[ "${LOG_WANDB}" == "false" ]]; then
        cmd="${cmd} logger.mode=disabled"
    fi

    echo -e "${BLUE}Submitting: ${description}${NC}"
    echo ""

    local sbatch_output
    sbatch_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/msillm_train_${job_suffix}_%j.out
#SBATCH --error=${LOG_DIR}/msillm_train_${job_suffix}_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=A100-80GB,H200
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source ~/.bashrc

# Fix git executable path issue
export PATH=/usr/bin:\$PATH
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet

# Set conda environment path
export PATH=\$HOME/anaconda3/envs/mode_env/bin:\$PATH
export CONDA_DEFAULT_ENV=mode_env
export CONDA_PREFIX=\$HOME/anaconda3/envs/mode_env

cd "${SCRIPT_DIR}"
${cmd}
EOF
    )
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Submitted: ${description}${NC}"
        echo -e "${GREEN}  ${sbatch_output}${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Submission failed: ${description}${NC}"
        echo -e "${RED}  ${sbatch_output}${NC}"
        echo ""
        return 1
    fi
}

# Track results
TOTAL=0
SUBMITTED=0
FAILED=0
FAILED_TESTS=()

# Define all training jobs to run
TRAININGS=()

# Add ALL MS-ILLM quality variants (vlo1, vlo2, 1-6)
# Uncomment the ones you want to train
TRAININGS+=("msillm_quality_vlo1|MS-ILLM Quality VLO1")
TRAININGS+=("msillm_quality_vlo2|MS-ILLM Quality VLO2")
TRAININGS+=("msillm_quality_1|MS-ILLM Quality 1")
TRAININGS+=("msillm_quality_2|MS-ILLM Quality 2")
TRAININGS+=("msillm_quality_3|MS-ILLM Quality 3")
TRAININGS+=("msillm_quality_4|MS-ILLM Quality 4")
TRAININGS+=("msillm_quality_5|MS-ILLM Quality 5")
TRAININGS+=("msillm_quality_6|MS-ILLM Quality 6")

TOTAL=${#TRAININGS[@]}

if [ ${TOTAL} -eq 0 ]; then
    echo -e "${YELLOW}Warning: No training jobs defined. Please uncomment at least one entry in TRAININGS array.${NC}"
    exit 1
fi

# Run all training jobs
for i in "${!TRAININGS[@]}"; do
    # Temporarily disable exit on error for array operations
    set +e
    IFS='|' read -r entrypoint description <<< "${TRAININGS[$i]}"
    set -e
    
    index=$((i + 1))
    
    # Temporarily disable exit on error for function call and arithmetic
    set +e
    if submit_training "${entrypoint}" "${description}" ${index} ${TOTAL}; then
        ((SUBMITTED++)) || true  # || true prevents exit on arithmetic failure
    else
        ((FAILED++)) || true  # || true prevents exit on arithmetic failure
        FAILED_TESTS+=("${description}")
    fi
    set -e
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Submission Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total: ${TOTAL}"
echo -e "${GREEN}Submitted: ${SUBMITTED}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed submissions: ${FAILED}${NC}"
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- ${test}${NC}"
    done
else
    echo -e "${GREEN}Failed submissions: 0${NC}"
fi
echo -e "${GREEN}========================================${NC}"

# Exit with error if any test failed
if [ ${FAILED} -gt 0 ]; then
    exit 1
fi


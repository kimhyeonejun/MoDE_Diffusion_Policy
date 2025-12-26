#!/bin/bash

# Submit SLURM jobs to evaluate all .ckpt files in saved_models directory.
# Each checkpoint is submitted as a separate sbatch job.
# Usage:
#   ./evaluate_all_checkpoints.sh
#   DEVICE=1 ./evaluate_all_checkpoints.sh          # Override GPU index
#   LOG_WANDB=false ./evaluate_aWll_checkpoints.sh   # Disable wandb

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/mode/evaluation/mode_evaluate_libero_msillm.py"
SAVED_MODELS_DIR="${SCRIPT_DIR}/saved_models"
DEVICE=${DEVICE:-0}
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
echo -e "${GREEN}Evaluate All Checkpoints Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Device: ${DEVICE}"
echo -e "Wandb logging: ${LOG_WANDB}"
echo ""

# Check if saved_models directory exists
if [ ! -d "${SAVED_MODELS_DIR}" ]; then
    echo -e "${RED}Error: saved_models directory not found at ${SAVED_MODELS_DIR}${NC}"
    exit 1
fi

# Submit evaluation as an sbatch job
submit_evaluation() {
    set +e
    local ckpt_filename=$1
    local description=$2
    local index=$3
    local total=$4

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[${index}/${total}] Submitting: ${description}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "Checkpoint: ${ckpt_filename}"
    echo ""

    # Sanitize job name from checkpoint filename
    local job_suffix="${ckpt_filename%.ckpt}"  # Remove .ckpt extension
    job_suffix=${job_suffix//[^A-Za-z0-9_.-]/-}  # Replace invalid chars with -
    local job_name="ckpt-eval-${job_suffix}"

    # Build command - use python from conda env directly
    local python_cmd="\${HOME}/anaconda3/envs/mode_env/bin/python"
    local cmd="${python_cmd} ${EVAL_SCRIPT} 'checkpoint=\"${ckpt_filename}\"' use_reconstructed_video=true"
    if [ "${LOG_WANDB}" != "true" ]; then
        cmd="${cmd} log_wandb=false"
    fi

    echo -e "${BLUE}Submitting: ${description}${NC}"
    echo ""

    local sbatch_output
    sbatch_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/ckpt_eval_${job_suffix}_%j.out
#SBATCH --error=${LOG_DIR}/ckpt_eval_${job_suffix}_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=RTX3090,RTX6000ADA,L40S,TITANRTX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

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

# Find all .ckpt files
echo -e "${BLUE}Searching for .ckpt files in ${SAVED_MODELS_DIR}...${NC}"
ckpt_files=($(find "${SAVED_MODELS_DIR}" -name "*.ckpt" -type f | sort))
total=${#ckpt_files[@]}

if [ ${total} -eq 0 ]; then
    echo -e "${YELLOW}No .ckpt files found in ${SAVED_MODELS_DIR}${NC}"
    exit 0
fi

echo -e "${GREEN}Found ${total} checkpoint file(s)${NC}"
echo ""

# Track results
TOTAL=${total}
SUBMITTED=0
FAILED=0
FAILED_FILES=()

# Process each checkpoint file
for i in "${!ckpt_files[@]}"; do
    ckpt_file="${ckpt_files[$i]}"
    ckpt_filename=$(basename "${ckpt_file}")
    index=$((i + 1))
    
    # Temporarily disable exit on error for function call and arithmetic
    set +e
    if submit_evaluation "${ckpt_filename}" "${ckpt_filename}" ${index} ${TOTAL}; then
        ((SUBMITTED++)) || true  # || true prevents exit on arithmetic failure
    else
        ((FAILED++)) || true  # || true prevents exit on arithmetic failure
        FAILED_FILES+=("${ckpt_filename}")
    fi
    set -e
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total checkpoints: ${TOTAL}"
echo -e "${GREEN}Submitted: ${SUBMITTED}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed submissions: ${FAILED}${NC}"
    echo -e "${RED}Failed files:${NC}"
    for file in "${FAILED_FILES[@]}"; do
        echo -e "  ${RED}- ${file}${NC}"
    done
else
    echo -e "${GREEN}Failed submissions: 0${NC}"
fi
echo -e "${GREEN}========================================${NC}"

# Exit with error if any test failed
if [ ${FAILED} -gt 0 ]; then
    exit 1
fi


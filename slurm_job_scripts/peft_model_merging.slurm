#!/bin/bash
#SBATCH --account=aip-wangcs
#SBATCH --job-name=peft_lora_model_merging  # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --error=%x_%j.err        # Standard error log
#SBATCH --nodes=1                  # total number of nodes
#SBATCH --gpus-per-node=h100:4      # number of GPUs per node
#SBATCH --ntasks-per-node=1        # Run 1 task (your script) on the node
#SBATCH --cpus-per-task=4          # Give that 1 task 4 CPU cores
#SBATCH --mem=32G                    # Memory
#SBATCH --time=0-00:15:00            # time (DD-HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SIF_PATH="${PROJECT_ROOT}/flow-factory.sif"
MERGING_SCRIPT="${PROJECT_ROOT}/peft_model_merge.py"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# Merge target settings (override via environment variables when submitting).
BASE_MODEL_ID="${BASE_MODEL_ID:-Wan-AI/Wan2.1-I2V-14B-720P-Diffusers}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/saves/wan2_i2v_lora_nft_20260322_130505/checkpoints/checkpoint-120}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAPTER_DIR}/full_model}"
OUTPUT_STYLE="${OUTPUT_STYLE:-weights}"  # diffusers | weights | both

export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1

module load apptainer

apptainer exec --nv \
  "${SIF_PATH}" \
  python3 "${MERGING_SCRIPT}" \
  --base-model-id "${BASE_MODEL_ID}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-style "${OUTPUT_STYLE}"

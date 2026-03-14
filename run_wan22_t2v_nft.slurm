#!/bin/bash
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --job-name=flow_factory_nft_wan22_t2v  # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --error=%x_%j.err        # Standard error log
#SBATCH --nodes=1                  # total number of nodes
#SBATCH --gpus-per-node=h200:8      # number of GPUs per node
#SBATCH --ntasks-per-node=1        # Run 1 task (your script) on the node
#SBATCH --cpus-per-task=16         # Give that 1 task 16 CPU cores (MATCHES filter_workers=16)
#SBATCH --mem=0                    # Use the whole node
#SBATCH --time=0-02:00:00            # time (DD-HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SIF_PATH="${PROJECT_ROOT}/flow-factory.sif"
CONFIG_REL="examples/nft/lora/wan22_t2v.yaml"

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "Error: SIF file not found at ${SIF_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/${CONFIG_REL}" ]]; then
  echo "Error: YAML config not found at ${PROJECT_ROOT}/${CONFIG_REL}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set. Export it before submitting with sbatch." >&2
  exit 1
fi

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN_RESOLVED}" ]]; then
  echo "Error: HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set. It is required for Hugging Face download/upload." >&2
  exit 1
fi

WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT:-Flow-Factory-WAN22-T2V-NFT}"
export APPTAINERENV_WANDB_MODE="${WANDB_MODE:-online}"
export APPTAINERENV_WANDB_DIR="${WANDB_DIR}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
fi

export APPTAINERENV_HF_TOKEN="${HF_TOKEN_RESOLVED}"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_RESOLVED}"
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"

echo "Starting NFT fine-tuning job"
echo "Project root: ${PROJECT_ROOT}"
echo "SIF: ${SIF_PATH}"
echo "Config: ${CONFIG_REL}"
echo "W&B project: ${APPTAINERENV_WANDB_PROJECT}"
echo "HF cache: ${HF_HOME}"

# NOTE: The YAML may set a fixed num_processes value.
# Override it with the GPU count allocated by SLURM for this job.
NUM_PROCESSES="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
if [[ "${NUM_PROCESSES}" == *:* ]]; then
  NUM_PROCESSES="${NUM_PROCESSES##*:}"
fi
NUM_PROCESSES="${NUM_PROCESSES%%[^0-9]*}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# Load apptainer module if required by the cluster
module load apptainer

echo "Using num_processes=${NUM_PROCESSES}"

srun --ntasks=1 apptainer exec --nv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${WANDB_DIR}:${WANDB_DIR}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --bind /dev/shm:/dev/shm \
  --bind /etc/ssl/certs:/etc/ssl/certs:ro \
  --bind /etc/pki:/etc/pki:ro \
  "${SIF_PATH}" \
  bash -lc "cd \"${PROJECT_ROOT}\" && ff-train \"${CONFIG_REL}\" --num_processes ${NUM_PROCESSES}"

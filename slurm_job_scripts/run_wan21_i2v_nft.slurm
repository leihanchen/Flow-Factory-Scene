#!/bin/bash
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --job-name=flow_factory_nft_wan21_i2v  # Job name
#SBATCH --output=%x_%j.out       # Standard output and error log
#SBATCH --error=%x_%j.err        # Standard error log
#SBATCH --nodes=1                  # total number of nodes
#SBATCH --gpus-per-node=h200:8      # number of GPUs per node
#SBATCH --ntasks-per-node=1        # Run 1 task (your script) on the node
#SBATCH --cpus-per-task=8          # Give that 1 task 8 CPU cores
#SBATCH --mem=0                    # Use the whole node
#SBATCH --time=0-02:00:00            # time (DD-HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SIF_PATH="${PROJECT_ROOT}/flow-factory.sif"
CONFIG_REL="examples/nft/lora/wan21_i2v.yaml"
CONTAINER_PROJECT_ROOT="/opt/Flow-Factory"

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "Error: SIF file not found at ${SIF_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/${CONFIG_REL}" ]]; then
  echo "Error: YAML config not found at ${PROJECT_ROOT}/${CONFIG_REL}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Export environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

OFFLINE_MODE="${OFFLINE_MODE:-1}"   # 1 for offline node, 0 for online
if [[ "${OFFLINE_MODE}" != "0" && "${OFFLINE_MODE}" != "1" ]]; then
  echo "Error: OFFLINE_MODE must be 0 or 1 (got '${OFFLINE_MODE}')." >&2
  exit 1
fi

HF_TOKEN_RESOLVED="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${HF_DATASETS_CACHE}}"
LOG_SAVE_DIR="${LOG_SAVE_DIR:-${PROJECT_ROOT}/outputs/wan22_t2v_nft}"

mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${DATA_CACHE_DIR}" "${LOG_SAVE_DIR}"

export APPTAINERENV_OFFLINE_MODE="${OFFLINE_MODE}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT:-Flow-Factory-WAN22-T2V-NFT}"
export APPTAINERENV_WANDB_DIR="${WANDB_DIR}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
fi

export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export APPTAINERENV_PYTHONPATH="${CONTAINER_PROJECT_ROOT}/src:${CONTAINER_PROJECT_ROOT}:${PYTHONPATH}"
else
  export APPTAINERENV_PYTHONPATH="${CONTAINER_PROJECT_ROOT}/src:${CONTAINER_PROJECT_ROOT}"
fi
export APPTAINERENV_PYTHONNOUSERSITE=1

if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export APPTAINERENV_WANDB_MODE="${WANDB_MODE:-offline}"
  export APPTAINERENV_HF_HUB_OFFLINE=1
  export APPTAINERENV_TRANSFORMERS_OFFLINE=1
  export APPTAINERENV_HF_DATASETS_OFFLINE=1

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
  fi

  if [[ -n "${HF_TOKEN_RESOLVED}" ]]; then
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN_RESOLVED}"
    export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_RESOLVED}"
  fi
else
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Error: WANDB_API_KEY is not set. Export it before submitting with OFFLINE_MODE=0." >&2
    exit 1
  fi

  if [[ -z "${HF_TOKEN_RESOLVED}" ]]; then
    echo "Error: HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set. It is required when OFFLINE_MODE=0." >&2
    exit 1
  fi

  export APPTAINERENV_WANDB_MODE="${WANDB_MODE:-online}"
  export APPTAINERENV_HF_HUB_OFFLINE=0
  export APPTAINERENV_TRANSFORMERS_OFFLINE=0
  export APPTAINERENV_HF_DATASETS_OFFLINE=0
  export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN_RESOLVED}"
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_RESOLVED}"
fi

echo "Starting NFT fine-tuning job"
echo "Project root: ${PROJECT_ROOT}"
echo "Container project root: ${CONTAINER_PROJECT_ROOT}"
echo "SIF: ${SIF_PATH}"
echo "Config: ${CONFIG_REL}"
echo "OFFLINE_MODE: ${OFFLINE_MODE}"
echo "W&B project: ${APPTAINERENV_WANDB_PROJECT}"
echo "W&B mode: ${APPTAINERENV_WANDB_MODE}"
echo "HF cache: ${HF_HOME}"
echo "HF hub offline: ${APPTAINERENV_HF_HUB_OFFLINE}"
echo "Container PYTHONPATH: ${APPTAINERENV_PYTHONPATH}"

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
  --bind "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
  --bind "${WANDB_DIR}:${WANDB_DIR}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --bind /dev/shm:/dev/shm \
  --bind /etc/ssl/certs:/etc/ssl/certs:ro \
  --bind /etc/pki:/etc/pki:ro \
  "${SIF_PATH}" \
  bash -lc "cd \"${CONTAINER_PROJECT_ROOT}\" && ff-train \"${CONFIG_REL}\" --num_processes ${NUM_PROCESSES}"

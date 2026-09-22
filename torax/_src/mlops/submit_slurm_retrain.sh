#!/usr/bin/env bash
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# SLURM submission script for TORAX active learning surrogate model retraining.
# ==============================================================================
#
#SBATCH --job-name=torax_retrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/torax_retrain_%j.log
#SBATCH --error=logs/torax_retrain_%j.err

set -euo pipefail

# Default arguments
HARVEST_DIR="${HARVEST_DIR:-/tmp/torax_harvest}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/torax_models}"
FINGERPRINT="${FINGERPRINT:-tglf_sat1}"
EPOCHS="${EPOCHS:-30}"
MAX_REGRESSION_PCT="${MAX_REGRESSION_PCT:-5.0}"

# Parse command line overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --harvest_dir)
      HARVEST_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --fingerprint)
      FINGERPRINT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --max_regression_pct)
      MAX_REGRESSION_PCT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "Starting TORAX Surrogate Retraining on SLURM Cluster"
echo "Job ID:             ${SLURM_JOB_ID:-local}"
echo "Node:               $(hostname)"
echo "Harvest Dir:        ${HARVEST_DIR}"
echo "Output Dir:         ${OUTPUT_DIR}"
echo "Solver Fingerprint: ${FINGERPRINT}"
echo "Epochs:             ${EPOCHS}"
echo "Start Time:         $(date)"
echo "=========================================================="

python3 -m torax._src.mlops.train_surrogate \
  --harvest_dir "${HARVEST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --fingerprint "${FINGERPRINT}" \
  --epochs "${EPOCHS}" \
  --max_regression_pct "${MAX_REGRESSION_PCT}"

echo "=========================================================="
echo "Retraining Job Completed Successfully: $(date)"
echo "=========================================================="

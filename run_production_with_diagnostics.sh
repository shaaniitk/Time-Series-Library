#!/bin/bash
################################################################################
# Production Training with Comprehensive Diagnostics
# 
# This script runs full production training with:
# - Gradient flow diagnostics
# - Attention weight logging
# - Component performance metrics
# - Memory profiling
# - Loss decomposition
# - Model checkpoint analysis
################################################################################

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/celestial_enhanced_pgat_production.yaml"
EXPERIMENT_NAME="prod_hierarchical_fusion_c2t_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}/gradients"
mkdir -p "${LOG_DIR}/attention"
mkdir -p "${LOG_DIR}/components"

echo "================================================================================================"
echo "Starting Production Training with Diagnostics"
echo "================================================================================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Logs: ${LOG_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "================================================================================================"

# Run training with comprehensive logging
python -X utf8 scripts/train/train_celestial_production.py \
    --config "${CONFIG_FILE}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --log_dir "${LOG_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --enable_gradient_diagnostics \
    --enable_attention_logging \
    --enable_component_profiling \
    --log_interval 10 \
    --checkpoint_interval 100 \
    --gradient_log_interval 50 \
    --attention_log_interval 100 \
    2>&1 | tee "${LOG_DIR}/training_full.log"

echo ""
echo "================================================================================================"
echo "Training Complete!"
echo "================================================================================================"
echo "Logs saved to: ${LOG_DIR}"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo ""
echo "To analyze diagnostics, run:"
echo "  python scripts/analysis/analyze_training_diagnostics.py --log_dir ${LOG_DIR}"
echo "================================================================================================"

#!/bin/bash
# Auto-generated training script
# Created: 2025-06-17 15:39:41
# Mode: Multi-target (MS)
# Complexity: Heavy
# Model: hierarchical

echo "🚀 Starting Enhanced Autoformer Training"
echo "Configuration: config_enhanced_autoformer_MS_heavy.yaml"
echo "Model Type: hierarchical"
echo "Estimated training time: 1-4 hours (GPU) / 2-12 hours (CPU)"

python train_configurable_autoformer.py \
    --config config_enhanced_autoformer_MS_heavy.yaml \
    --model_type hierarchical

echo "✅ Training completed!"

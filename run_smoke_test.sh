#!/bin/bash
# Quick smoke test for Phase 1 MDN decoder
# Runs 2 epochs with minimal config

echo "========================================================================"
echo "🔥 Phase 1 MDN Decoder - Smoke Test"
echo "========================================================================"
echo ""
echo "Configuration: test_celestial_smoke.yaml"
echo "  - 2 epochs only"
echo "  - Minimal dimensions (d_model=64)"
echo "  - MDN enabled (K=3 components)"
echo "  - Fast validation (not convergence)"
echo ""
echo "Starting training..."
echo "========================================================================"
echo ""

cd "$(dirname "$0")/.." || exit 1

python scripts/train/train_celestial_production.py \
    --config configs/test_celestial_smoke.yaml \
    --model Celestial_Enhanced_PGAT \
    --data custom

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Smoke test completed successfully!"
    echo ""
    echo "Check results:"
    echo "  - Training logs: logs/*.log"
    echo "  - MDN calibration: grep 'MDN CALIBRATION' logs/memory_diagnostics_*.log"
    echo "  - Checkpoints: checkpoints/*test_celestial_smoke*"
else
    echo "❌ Smoke test failed with exit code $EXIT_CODE"
    echo ""
    echo "Check error logs for details"
fi
echo "========================================================================"

exit $EXIT_CODE

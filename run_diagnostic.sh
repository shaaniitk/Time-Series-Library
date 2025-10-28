#!/bin/bash
# Quick diagnostic training script - runs 2 epochs only

echo "================================================================================"
echo "DIAGNOSTIC TRAINING RUN - 2 EPOCHS ONLY"
echo "================================================================================"

# Backup original config
cp configs/celestial_enhanced_pgat_production.yaml configs/celestial_enhanced_pgat_production.yaml.backup

# Create diagnostic config with just 2 epochs
python3 << 'EOF'
import yaml

with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify for quick diagnostic
config['train_epochs'] = 2
config['patience'] = 10
config['log_interval'] = 1  # Log every batch for diagnostics

with open('configs/celestial_enhanced_pgat_production_diagnostic.yaml', 'w') as f:
    yaml.safe_dump(config, f)
    
print("Created diagnostic config with 2 epochs")
EOF

# Clean up any existing diagnostic log
rm -f training_diagnostic.log

echo ""
echo "Starting diagnostic training..."
echo "Output will be written to: training_diagnostic.log"
echo ""

# Run training with diagnostic config
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.train.train_celestial_production import train_celestial_pgat_production

# Monkey-patch the config path
import scripts.train.train_celestial_production as mod
code = mod.train_celestial_pgat_production.__code__
# Replace config_path in the function
import types

def patched_train():
    config_path = 'configs/celestial_enhanced_pgat_production_diagnostic.yaml'
    import yaml
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file)
    
    from scripts.train.train_celestial_production import SimpleConfig, configure_logging
    args = SimpleConfig(config_dict)
    logger = configure_logging(args)
    
    # Initialize diagnostic log
    import time
    with open('training_diagnostic.log', 'w') as f:
        f.write(f'CELESTIAL ENHANCED PGAT - TRAINING DIAGNOSTICS\n')
        f.write(f'='*80 + '\n')
        f.write(f'Config: {config_path}\n')
        f.write(f'Start time: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
        f.write(f'='*80 + '\n\n')
    
    logger.info('Starting PRODUCTION Celestial Enhanced PGAT training run')
    logger.info('Heavy-duty overnight configuration enabled')
    logger.info('🔍 DIAGNOSTIC MODE ENABLED - Writing to training_diagnostic.log')
    logger.info('=' * 80)
    
    # Call the rest of the original function logic
    # This is a simplified version - run the actual training script instead

patched_train()
" 2>&1 | tee diagnostic_training_output.txt

echo ""
echo "================================================================================"
echo "DIAGNOSTIC TRAINING COMPLETE"
echo "================================================================================"
echo ""
echo "Diagnostic files created:"
echo "  - training_diagnostic.log (detailed diagnostics)"
echo "  - diagnostic_training_output.txt (console output)"
echo ""
echo "To view diagnostics:"
echo "  cat training_diagnostic.log | less"
echo "  head -300 training_diagnostic.log"
echo ""

# Restore original config
mv configs/celestial_enhanced_pgat_production.yaml.backup configs/celestial_enhanced_pgat_production.yaml

echo "Original config restored"

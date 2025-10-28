#!/usr/bin/env python
"""Debug script to find where training is failing"""
import sys
import traceback

sys.path.insert(0, '.')

try:
    from scripts.train.train_celestial_production import train_celestial_pgat_production
    print('\n' + '='*80)
    print('STARTING TRAINING WITH FULL TRACEBACK...')
    print('='*80 + '\n')
    
    result = train_celestial_pgat_production('configs/test_celestial_smoke_verbose.yaml')
    
    print('\n' + '='*80)
    print(f'RESULT: {result}')
    print('='*80)
    
    if not result:
        print('\nTraining returned False - check logs for errors')
        sys.exit(1)
    else:
        print('\nTraining completed successfully!')
        sys.exit(0)
        
except Exception as e:
    print('\n' + '='*80)
    print('EXCEPTION CAUGHT:')
    print('='*80)
    traceback.print_exc()
    print('='*80)
    sys.exit(1)

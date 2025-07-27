#!/usr/bin/env python3

import sys
import argparse
from run import get_parser, fix_seed

# Force CPU usage
def main():
    parser = get_parser()
    
    # Simulate command line args
    sys.argv = [
        'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './dataset/synthetic_time_series/',
        '--data_path', 'synthetic_timeseries.csv',
        '--model_id', 'test_debug',
        '--model', 'HierarchicalEnhancedAutoformer',
        '--data', 'MS',
        '--features', 'M',
        '--seq_len', '96',
        '--label_len', '48',
        '--pred_len', '24',
        '--e_layers', '2',
        '--d_layers', '1',
        '--factor', '3',
        '--enc_in', '7',
        '--dec_in', '7',
        '--c_out', '4',
        '--des', 'Experiment',
        '--d_model', '32',
        '--d_ff', '128',
        '--train_epochs', '2',
        '--batch_size', '16'
    ]
    
    args = parser.parse_args()
    
    # Force CPU usage
    args.use_gpu = False
    
    print('Args set: use_gpu =', args.use_gpu)
    
    # Import after args are set
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    # Fix random seed for reproducibility
    fix_seed(args.random_seed)

    print(f'Args in experiment:')
    print(f'Task Name: {args.task_name}')
    print(f'Model: {args.model}')
    print(f'Use GPU: {args.use_gpu}')
    print('')

    # Set up experiment
    exp = Exp_Long_Term_Forecast(args)
    
    # Training
    setting = f"{args.task_name}_{args.model_id}_{args.model}_debug"
    exp.train(setting)

if __name__ == '__main__':
    main()

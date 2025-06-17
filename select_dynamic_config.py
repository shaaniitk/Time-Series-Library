#!/usr/bin/env python3
"""
Dynamic Enhanced Autoformer Configuration Selector

Interactive script that automatically analyzes your dataset and generates
appropriate configurations based on your data dimensions and requirements.
"""

import os
import sys
import json
import yaml
from datetime import datetime
from utils.data_analysis import analyze_dataset, generate_dynamic_config, validate_config_with_data

def display_banner():
    """Display welcome banner"""
    print("🎯 Dynamic Enhanced Autoformer Configuration Selector")
    print("=" * 70)
    print("This tool analyzes your dataset and creates personalized configurations.")
    print("No more hardcoded dimensions - everything adapts to YOUR data!")
    print("=" * 70)

def get_dataset_info():
    """Get dataset information from user"""
    print("\n📂 DATASET CONFIGURATION")
    print("-" * 30)
    
    # Get data path
    while True:
        data_path = input("Enter path to your CSV data file: ").strip()
        if os.path.exists(data_path):
            break
        print(f"❌ File not found: {data_path}")
        print("Please enter a valid file path.")
    
    # Get target columns (optional)
    print("\n🎯 TARGET COLUMNS")
    print("Specify your target columns (the variables you want to predict).")
    print("Leave empty to auto-detect OHLC columns (first 4 non-date columns).")
    
    target_input = input("Target columns (comma-separated, or press Enter for auto): ").strip()
    target_columns = None if not target_input else target_input
    
    return data_path, target_columns

def analyze_user_dataset(data_path, target_columns):
    """Analyze user's dataset and display results"""
    print(f"\n🔍 ANALYZING YOUR DATASET")
    print("-" * 30)
    print("Scanning data structure and dimensions...")
    
    try:
        analysis = analyze_dataset(data_path, target_columns)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Dataset: {os.path.basename(data_path)}")
        print(f"📈 Samples: {analysis['n_samples']:,}")
        print(f"🎯 Target features: {analysis['n_targets']} {analysis['target_columns']}")
        print(f"📋 Covariate features: {analysis['n_covariates']}")
        print(f"📌 Total features: {analysis['n_total_features']}")
        
        print(f"\n🎛️ Available Forecasting Modes:")
        for mode in ['M', 'MS', 'S']:
            mode_config = analysis[f'mode_{mode}']
            print(f"  {mode}: {mode_config['description']}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

def get_complexity_requirements():
    """Get user's computational requirements"""
    print("\n💻 COMPUTATIONAL RESOURCES")
    print("-" * 30)
    
    questions = [
        {
            'question': "What's your primary hardware?",
            'options': [
                ("1", "CPU only (no GPU)", "cpu"),
                ("2", "GPU with 4-8GB VRAM", "gpu_small"),
                ("3", "GPU with 8GB+ VRAM", "gpu_large"),
                ("4", "High-end workstation/server", "workstation")
            ]
        },
        {
            'question': "How much RAM do you have available?",
            'options': [
                ("1", "4-8GB", "ram_low"),
                ("2", "8-16GB", "ram_medium"),
                ("3", "16-32GB", "ram_high"),
                ("4", "32GB+", "ram_very_high")
            ]
        },
        {
            'question': "How much time can you spend on training?",
            'options': [
                ("1", "5-15 minutes (quick prototype)", "time_minimal"),
                ("2", "15-60 minutes (development)", "time_short"),
                ("3", "1-4 hours (research)", "time_medium"),
                ("4", "4+ hours (production)", "time_long")
            ]
        }
    ]
    
    answers = {}
    for q in questions:
        print(f"\n{q['question']}")
        for option in q['options']:
            print(f"  {option[0]}. {option[1]}")
        
        while True:
            choice = input("\nYour choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                answers[q['question']] = q['options'][int(choice)-1][2]
                break
            print("Please enter 1, 2, 3, or 4")
    
    return answers

def get_forecasting_requirements():
    """Get user's forecasting requirements"""
    print("\n🎯 FORECASTING REQUIREMENTS")
    print("-" * 30)
    
    questions = [
        {
            'question': "What type of forecasting do you need?",
            'options': [
                ("1", "Full multivariate forecasting (predict all features)", "M"),
                ("2", "Multi-target prediction (use all features to predict targets)", "MS"),
                ("3", "Target-only forecasting (use and predict only targets)", "S")
            ]
        },
        {
            'question': "What's your priority?",
            'options': [
                ("1", "Speed and efficiency", "speed"),
                ("2", "Balanced performance", "balanced"),
                ("3", "Maximum accuracy", "accuracy")
            ]
        },
        {
            'question': "What's your use case?",
            'options': [
                ("1", "Quick prototyping/experimentation", "prototype"),
                ("2", "Research and development", "research"),
                ("3", "Production deployment", "production")
            ]
        }
    ]
    
    answers = {}
    for q in questions:
        print(f"\n{q['question']}")
        for option in q['options']:
            print(f"  {option[0]}. {option[1]}")
        
        while True:
            choice = input("\nYour choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                answers[q['question']] = q['options'][int(choice)-1][2]
                break
            print("Please enter 1, 2, or 3")
    
    return answers

def recommend_configuration(comp_answers, forecast_answers):
    """Recommend configuration based on answers"""
    
    # Extract key factors
    hardware = comp_answers["What's your primary hardware?"]
    ram = comp_answers["How much RAM do you have available?"]
    time = comp_answers["How much time can you spend on training?"]
    
    mode = forecast_answers["What type of forecasting do you need?"]
    priority = forecast_answers["What's your priority?"]
    use_case = forecast_answers["What's your use case?"]
    
    # Determine complexity level
    complexity_score = 0
    
    # Hardware scoring
    if hardware == "cpu":
        complexity_score += 0
    elif hardware == "gpu_small":
        complexity_score += 2
    elif hardware == "gpu_large":
        complexity_score += 3
    else:  # workstation
        complexity_score += 4
    
    # RAM scoring
    if ram == "ram_low":
        complexity_score += 0
    elif ram == "ram_medium":
        complexity_score += 1
    elif ram == "ram_high":
        complexity_score += 2
    else:  # ram_very_high
        complexity_score += 3
    
    # Time scoring
    if time == "time_minimal":
        complexity_score += 0
    elif time == "time_short":
        complexity_score += 1
    elif time == "time_medium":
        complexity_score += 2
    else:  # time_long
        complexity_score += 3
    
    # Priority adjustment
    if priority == "speed":
        complexity_score -= 1
    elif priority == "accuracy":
        complexity_score += 2
    
    # Use case adjustment
    if use_case == "prototype":
        complexity_score -= 1
    elif use_case == "production":
        complexity_score += 1
    
    # Ensure score is within bounds
    complexity_score = max(0, min(10, complexity_score))
    
    # Map score to complexity level
    if complexity_score <= 2:
        complexity = "ultralight"
    elif complexity_score <= 4:
        complexity = "light"
    elif complexity_score <= 6:
        complexity = "medium"
    elif complexity_score <= 8:
        complexity = "heavy"
    else:
        complexity = "veryheavy"
    
    return mode, complexity, complexity_score

def get_configuration_info(mode, complexity):
    """Get detailed information about the recommended configuration"""
    
    config_map = {
        "ultralight": {
            "name": "Ultra-Light",
            "d_model": 32,
            "layers": "1+1",
            "seq_len": 50,
            "batch_size": 64,
            "features": "Basic only",
            "time_cpu": "3-10 min",
            "time_gpu": "1-3 min",
            "ram": "2-4GB"
        },
        "light": {
            "name": "Light",
            "d_model": 64,
            "layers": "2+1", 
            "seq_len": 100,
            "batch_size": 48,
            "features": "Adaptive correlation",
            "time_cpu": "10-20 min",
            "time_gpu": "3-8 min",
            "ram": "4-8GB"
        },
        "medium": {
            "name": "Medium",
            "d_model": 128,
            "layers": "3+2",
            "seq_len": 250,
            "batch_size": 32,
            "features": "Multi-scale + learnable decomp",
            "time_cpu": "30-90 min",
            "time_gpu": "10-30 min",
            "ram": "6-12GB"
        },
        "heavy": {
            "name": "Heavy",
            "d_model": 256,
            "layers": "4+3",
            "seq_len": 400,
            "batch_size": 16,
            "features": "All advanced + curriculum",
            "time_cpu": "2-12 hours",
            "time_gpu": "1-4 hours",
            "ram": "8-16GB"
        },
        "veryheavy": {
            "name": "Very Heavy",
            "d_model": 512,
            "layers": "6+4",
            "seq_len": 500,
            "batch_size": 8,
            "features": "All features + uncertainty",
            "time_cpu": "6-48 hours",
            "time_gpu": "3-12 hours",
            "ram": "12-32GB"
        }
    }
    
    mode_map = {
        "M": {
            "name": "Multivariate (M)",
            "use_case": "Full multivariate forecasting"
        },
        "MS": {
            "name": "Multi-target (MS)",
            "use_case": "Target prediction with rich context"
        },
        "S": {
            "name": "Target-only (S)",
            "use_case": "Pure target analysis"
        }
    }
    
    return config_map[complexity], mode_map[mode]

def generate_dynamic_configs(data_analysis, mode, complexity):
    """Generate dynamic configuration files for the user's dataset"""
    
    # Find the base template config
    template_config = f"config_enhanced_autoformer_{mode}_{complexity}.yaml"
    
    if not os.path.exists(template_config):
        print(f"❌ Template config not found: {template_config}")
        return None
    
    # Generate dynamic config
    try:
        output_config = f"config_enhanced_autoformer_{mode}_{complexity}_dynamic.yaml"
        dynamic_config = generate_dynamic_config(
            template_config, 
            data_analysis, 
            output_config, 
            mode
        )
        
        print(f"✅ Generated dynamic config: {dynamic_config}")
        return dynamic_config
        
    except Exception as e:
        print(f"❌ Error generating dynamic config: {e}")
        return None

def display_recommendation(mode, complexity, score, comp_info, mode_info, data_analysis, config_file):
    """Display the final recommendation"""
    
    print(f"\n🎯 PERSONALIZED CONFIGURATION")
    print("=" * 70)
    
    # Dataset information
    print(f"📊 Your Dataset:")
    print(f"   File: {os.path.basename(data_analysis['data_path'])}")
    print(f"   Features: {data_analysis['n_total_features']} total ({data_analysis['n_targets']} targets + {data_analysis['n_covariates']} covariates)")
    print(f"   Targets: {', '.join(data_analysis['target_columns'])}")
    print(f"   Samples: {data_analysis['n_samples']:,}")
    
    # Mode information  
    mode_config = data_analysis[f'mode_{mode}']
    print(f"\n🎯 Mode: {mode_info['name']}")
    print(f"   Architecture: {mode_config['description']}")
    print(f"   Input dims: {mode_config['enc_in']}")
    print(f"   Output dims: {mode_config['c_out']}")
    print(f"   Use case: {mode_info['use_case']}")
    
    # Complexity information
    print(f"\n⚙️ Complexity: {comp_info['name']}")
    print(f"   Model dimension: {comp_info['d_model']}")
    print(f"   Layers (enc+dec): {comp_info['layers']}")
    print(f"   Sequence length: {comp_info['seq_len']}")
    print(f"   Batch size: {comp_info['batch_size']}")
    print(f"   Enhanced features: {comp_info['features']}")
    
    # Resource requirements
    print(f"\n📈 Resource Requirements:")
    print(f"   RAM needed: {comp_info['ram']}")
    print(f"   Training time (CPU): {comp_info['time_cpu']}")
    print(f"   Training time (GPU): {comp_info['time_gpu']}")
    
    # Configuration file
    print(f"\n📁 Dynamic Configuration: {config_file}")
    if os.path.exists(config_file):
        print(f"   ✅ File generated and ready to use")
    else:
        print(f"   ❌ File generation failed")
    
    return config_file

def get_model_type_choice():
    """Get user's choice of model type"""
    print(f"\n🤖 MODEL TYPE SELECTION")
    print("-" * 30)
    print("Which Enhanced Autoformer variant would you like to use?")
    print("  1. Enhanced Autoformer (adaptive features)")
    print("  2. Bayesian Enhanced Autoformer (uncertainty quantification)")
    print("  3. Hierarchical Enhanced Autoformer (wavelet processing)")
    
    while True:
        choice = input("\nYour choice (1-3): ").strip()
        if choice == "1":
            return "enhanced"
        elif choice == "2":
            return "bayesian"
        elif choice == "3":
            return "hierarchical"
        else:
            print("Please enter 1, 2, or 3")

def validate_and_show_existing_configs(data_analysis):
    """Show validation results for existing config files"""
    print(f"\n🔍 VALIDATING EXISTING CONFIGURATIONS")
    print("-" * 50)
    
    config_files = []
    for file in os.listdir('.'):
        if file.startswith('config_enhanced_autoformer_') and file.endswith('.yaml'):
            config_files.append(file)
    
    if not config_files:
        print("No existing configuration files found.")
        return
    
    valid_configs = []
    invalid_configs = []
    
    for config_file in config_files:
        try:
            validation = validate_config_with_data(config_file, data_analysis['data_path'])
            
            if validation['valid']:
                valid_configs.append(config_file)
                print(f"✅ {config_file}")
            else:
                invalid_configs.append((config_file, validation))
                print(f"❌ {config_file}")
                for issue in validation['issues']:
                    print(f"   Issue: {issue['description']}")
        
        except Exception as e:
            print(f"⚠️ {config_file} - Error: {e}")
    
    print(f"\nSummary: {len(valid_configs)} valid, {len(invalid_configs)} invalid configs")
    
    if invalid_configs:
        print(f"\n💡 Tip: Use the dynamic config generator to fix dimension mismatches!")

def main():
    """Main function"""
    display_banner()
    
    # Get dataset information
    data_path, target_columns = get_dataset_info()
    
    # Analyze dataset
    data_analysis = analyze_user_dataset(data_path, target_columns)
    if data_analysis is None:
        print("❌ Cannot proceed without valid dataset analysis.")
        return
    
    # Option to validate existing configs or create new one
    print("\nWhat would you like to do?")
    print("  1. Create personalized configuration (recommended)")
    print("  2. Validate existing configurations against your data")
    print("  3. Both - validate existing and create new")
    
    while True:
        choice = input("\nYour choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    if choice in ['2', '3']:
        validate_and_show_existing_configs(data_analysis)
        if choice == '2':
            return
    
    # Get user requirements for new config
    comp_answers = get_complexity_requirements()
    forecast_answers = get_forecasting_requirements()
    
    # Get recommendation
    mode, complexity, score = recommend_configuration(comp_answers, forecast_answers)
    comp_info, mode_info = get_configuration_info(mode, complexity)
    
    # Generate dynamic configuration
    config_file = generate_dynamic_configs(data_analysis, mode, complexity)
    if config_file is None:
        print("❌ Failed to generate dynamic configuration.")
        return
    
    # Display recommendation
    display_recommendation(mode, complexity, score, comp_info, mode_info, data_analysis, config_file)
    
    # Get model type choice
    model_type = get_model_type_choice()
    
    # Final summary
    print(f"\n🎉 FINAL CONFIGURATION")
    print("=" * 50)
    print(f"Dataset: {os.path.basename(data_analysis['data_path'])}")
    print(f"Mode: {mode} ({data_analysis[f'mode_{mode}']['description']})")
    print(f"Complexity: {complexity}")
    print(f"Model Type: {model_type}")
    print(f"Configuration: {config_file}")
    
    print(f"\n🚀 Training Command:")
    print(f"  python train_configurable_autoformer.py \\")
    print(f"      --config {config_file} \\")
    print(f"      --model_type {model_type}")
    
    # Optionally create a training script
    print(f"\n💡 Create a custom training script? (y/n): ", end="")
    create_script = input().strip().lower()
    
    if create_script in ['y', 'yes']:
        script_name = f"train_dynamic_{mode}_{complexity}_{model_type}.sh"
        script_content = f"""#!/bin/bash
# Auto-generated dynamic training script
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Dataset: {os.path.basename(data_analysis['data_path'])}
# Features: {data_analysis['n_total_features']} total ({data_analysis['n_targets']} targets + {data_analysis['n_covariates']} covariates)
# Mode: {mode} ({data_analysis[f'mode_{mode}']['description']})
# Complexity: {complexity} ({comp_info['name']})
# Model: {model_type}

echo "🚀 Starting Dynamic Enhanced Autoformer Training"
echo "Dataset: {os.path.basename(data_analysis['data_path'])}"
echo "Configuration: {config_file}"
echo "Model Type: {model_type}"
echo "Architecture: {data_analysis[f'mode_{mode}']['description']}"
echo "Estimated training time: {comp_info['time_gpu']} (GPU) / {comp_info['time_cpu']} (CPU)"

python train_configurable_autoformer.py \\
    --config {config_file} \\
    --model_type {model_type}

echo "✅ Training completed!"
"""
        
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_name, 0o755)
        
        print(f"✅ Created training script: {script_name}")
        print(f"   Run with: ./{script_name}")
    
    print(f"\n🎯 Happy training with your personalized Enhanced Autoformer!")
    print(f"   Your config automatically adapts to {data_analysis['n_total_features']} features!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Configuration selection cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

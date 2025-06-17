#!/usr/bin/env python3
"""
Enhanced Autoformer Configuration Selector

Interactive script to help users select the appropriate configuration
based on their requirements, resources, and use case.
"""

import os
import sys
import json
from datetime import datetime

def display_banner():
    """Display welcome banner"""
    print("🎯 Enhanced Autoformer Configuration Selector")
    print("=" * 60)
    print("This tool helps you choose the right configuration for your needs.")
    print("We'll ask a few questions about your requirements and resources.")
    print("=" * 60)

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
                ("1", "Full market forecasting (all 118 features)", "M"),
                ("2", "Price prediction with market context (OHLC with all inputs)", "MS"),
                ("3", "Pure price forecasting (OHLC only)", "S")
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
            "input": "All 118 features",
            "output": "All 118 features",
            "use_case": "Full market ecosystem forecasting"
        },
        "MS": {
            "name": "Multi-target (MS)",
            "input": "All 118 features", 
            "output": "4 target features (OHLC)",
            "use_case": "Price prediction with market context"
        },
        "S": {
            "name": "Target-only (S)",
            "input": "4 target features (OHLC)",
            "output": "4 target features (OHLC)",
            "use_case": "Pure technical analysis"
        }
    }
    
    return config_map[complexity], mode_map[mode]

def display_recommendation(mode, complexity, score, comp_info, mode_info):
    """Display the final recommendation"""
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION")
    print("=" * 60)
    
    print(f"📊 Mode: {mode_info['name']}")
    print(f"   Input: {mode_info['input']}")
    print(f"   Output: {mode_info['output']}")
    print(f"   Use case: {mode_info['use_case']}")
    
    print(f"\n⚙️ Complexity: {comp_info['name']}")
    print(f"   Model dimension: {comp_info['d_model']}")
    print(f"   Layers (enc+dec): {comp_info['layers']}")
    print(f"   Sequence length: {comp_info['seq_len']}")
    print(f"   Batch size: {comp_info['batch_size']}")
    print(f"   Enhanced features: {comp_info['features']}")
    
    print(f"\n📈 Resource Requirements:")
    print(f"   RAM needed: {comp_info['ram']}")
    print(f"   Training time (CPU): {comp_info['time_cpu']}")
    print(f"   Training time (GPU): {comp_info['time_gpu']}")
    
    config_file = f"config_enhanced_autoformer_{mode}_{complexity}.yaml"
    print(f"\n📁 Configuration File: {config_file}")
    
    if os.path.exists(config_file):
        print(f"   ✅ File exists and ready to use")
    else:
        print(f"   ❌ File not found - please check file name")
    
    print(f"\n🚀 Training Command:")
    print(f"   python train_configurable_autoformer.py \\")
    print(f"       --config {config_file} \\")
    print(f"       --model_type enhanced")
    
    return config_file

def show_all_configurations():
    """Show all available configurations"""
    
    print(f"\n📚 ALL AVAILABLE CONFIGURATIONS")
    print("=" * 80)
    
    modes = ["M", "MS", "S"]
    complexities = ["ultralight", "light", "medium", "heavy", "veryheavy"]
    
    print(f"{'Mode':<4} {'Complexity':<12} {'Config File':<45} {'Exists':<6}")
    print("-" * 80)
    
    for mode in modes:
        for complexity in complexities:
            config_file = f"config_enhanced_autoformer_{mode}_{complexity}.yaml"
            exists = "✅" if os.path.exists(config_file) else "❌"
            print(f"{mode:<4} {complexity:<12} {config_file:<45} {exists:<6}")
    
    print(f"\nTotal: {len(modes) * len(complexities)} configurations")

def get_model_type_choice():
    """Get user's choice of model type"""
    print(f"\n🤖 MODEL TYPE SELECTION")
    print("-" * 30)
    print("Which Enhanced Autoformer variant would you like to use?")
    print("  1. Enhanced Autoformer (standard enhanced features)")
    print("  2. Bayesian Enhanced Autoformer (with uncertainty quantification)")
    print("  3. Hierarchical Enhanced Autoformer (with wavelet processing)")
    
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

def main():
    """Main function"""
    display_banner()
    
    # Option to show all configs or get recommendation
    print("\nWhat would you like to do?")
    print("  1. Get personalized configuration recommendation")
    print("  2. View all available configurations")
    
    while True:
        choice = input("\nYour choice (1-2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    if choice == '2':
        show_all_configurations()
        return
    
    # Get user requirements
    comp_answers = get_complexity_requirements()
    forecast_answers = get_forecasting_requirements()
    
    # Get recommendation
    mode, complexity, score = recommend_configuration(comp_answers, forecast_answers)
    comp_info, mode_info = get_configuration_info(mode, complexity)
    
    # Display recommendation
    config_file = display_recommendation(mode, complexity, score, comp_info, mode_info)
    
    # Get model type choice
    model_type = get_model_type_choice()
    
    # Final summary
    print(f"\n🎉 FINAL SELECTION")
    print("=" * 40)
    print(f"Configuration: {config_file}")
    print(f"Model Type: {model_type}")
    print(f"Complete Command:")
    print(f"  python train_configurable_autoformer.py \\")
    print(f"      --config {config_file} \\")
    print(f"      --model_type {model_type}")
    
    # Optionally create a training script
    print(f"\n💡 Would you like to create a custom training script? (y/n): ", end="")
    create_script = input().strip().lower()
    
    if create_script in ['y', 'yes']:
        script_name = f"train_{mode}_{complexity}_{model_type}.sh"
        script_content = f"""#!/bin/bash
# Auto-generated training script
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Mode: {mode_info['name']}
# Complexity: {comp_info['name']}
# Model: {model_type}

echo "🚀 Starting Enhanced Autoformer Training"
echo "Configuration: {config_file}"
echo "Model Type: {model_type}"
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
    
    print(f"\n🎯 Happy training with Enhanced Autoformer!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Configuration selection cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

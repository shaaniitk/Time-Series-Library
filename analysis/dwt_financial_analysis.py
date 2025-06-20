#!/usr/bin/env python3
"""
Enhanced DWT Analysis for Financial Time Series Data

This script performs Discrete Wavelet Transform (DWT) analysis on financial data,
comparing target series with planetary sine wave components.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_financial_data(filepath: str) -> pd.DataFrame:
    """Load the prepared financial data"""
    print(f"📊 Loading financial data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Data shape: {df.shape}")
    print(f"   Columns: {list(df.columns[:10])}...") # Show first 10 columns
    return df

def apply_dwt_decomposition(signal: np.array, wavelet: str = 'db4', levels: int = 5) -> Dict:
    """
    Apply DWT decomposition and return coefficients
    
    Args:
        signal: Input time series signal
        wavelet: Wavelet type (default: 'db4')
        levels: Number of decomposition levels
        
    Returns:
        Dictionary containing decomposition coefficients
    """
    # Perform multilevel DWT
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    # Reconstruct individual components
    components = {}
    
    # Approximation (low frequency) - Level 5
    approx_coeffs = coeffs[0]
    components[f'A{levels}'] = pywt.upcoef('a', approx_coeffs, wavelet, level=levels, take=len(signal))
    
    # Details (high frequency) - Levels 5 to 1
    for i in range(1, levels + 1):
        detail_coeffs = coeffs[i]
        level = levels - i + 1
        components[f'D{level}'] = pywt.upcoef('d', detail_coeffs, wavelet, level=level, take=len(signal))
    
    return components, coeffs

def analyze_dwt_components(df: pd.DataFrame, target_columns: List[str], 
                          planetary_columns: List[str], num_components: int = 5) -> Dict:
    """
    Analyze DWT components for financial targets and planetary influences
    """
    print(f"\n🔬 Analyzing DWT components...")
    print(f"   Target columns: {target_columns}")
    print(f"   Planetary columns: {planetary_columns}")
    print(f"   Number of components to analyze: {num_components}")
    
    results = {}
    
    # Analyze target series
    print(f"\n📈 Analyzing Financial Targets:")
    for col in target_columns:
        if col in df.columns:
            signal = df[col].values
            components, coeffs = apply_dwt_decomposition(signal, levels=num_components)
            results[col] = {
                'signal': signal,
                'components': components,
                'coefficients': coeffs,
                'total_components': len(coeffs)  # 1 approximation + num_components details
            }
            print(f"   ✅ {col}: {len(coeffs)} total components (1 approx + {len(coeffs)-1} details)")
        else:
            print(f"   ❌ {col}: Column not found in data")
    
    # Analyze planetary series
    print(f"\n🌟 Analyzing Planetary Sine Waves:")
    for col in planetary_columns:
        if col in df.columns:
            signal = df[col].values
            components, coeffs = apply_dwt_decomposition(signal, levels=num_components)
            results[col] = {
                'signal': signal,
                'components': components,
                'coefficients': coeffs,
                'total_components': len(coeffs)
            }
            print(f"   ✅ {col}: {len(coeffs)} total components (1 approx + {len(coeffs)-1} details)")
        else:
            print(f"   ❌ {col}: Column not found in data")
    
    return results

def create_dwt_comparison_plots(results: Dict, target_columns: List[str], 
                               planetary_columns: List[str], num_components: int = 5):
    """Create comprehensive DWT comparison plots"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Original signals comparison
    plt.subplot(3, 2, 1)
    for col in target_columns:
        if col in results:
            plt.plot(results[col]['signal'][:500], label=f'{col}', alpha=0.8, linewidth=1.5)
    plt.title('Financial Target Series (First 500 points)', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    for col in planetary_columns:
        if col in results:
            plt.plot(results[col]['signal'][:500], label=f'{col}', alpha=0.8, linewidth=1.5)
    plt.title('Planetary Sine Wave Series (First 500 points)', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: DWT Components for first target
    if target_columns[0] in results:
        target_name = target_columns[0]
        components = results[target_name]['components']
        
        plt.subplot(3, 2, 3)
        plt.plot(components[f'A{num_components}'][:500], label=f'A{num_components} (Trend)', linewidth=2)
        for i in range(num_components, 0, -1):
            plt.plot(components[f'D{i}'][:500], label=f'D{i}', alpha=0.7)
        plt.title(f'DWT Components: {target_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: DWT Components for first planetary series
    if planetary_columns[0] in results:
        planet_name = planetary_columns[0]
        components = results[planet_name]['components']
        
        plt.subplot(3, 2, 4)
        plt.plot(components[f'A{num_components}'][:500], label=f'A{num_components} (Trend)', linewidth=2)
        for i in range(num_components, 0, -1):
            plt.plot(components[f'D{i}'][:500], label=f'D{i}', alpha=0.7)
        plt.title(f'DWT Components: {planet_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Energy distribution comparison
    plt.subplot(3, 2, 5)
    energy_data = []
    series_names = []
    
    for col in target_columns + planetary_columns:
        if col in results:
            components = results[col]['components']
            energies = []
            comp_names = []
            
            # Calculate energy for each component
            for comp_name, comp_signal in components.items():
                energy = np.sum(comp_signal**2)
                energies.append(energy)
                comp_names.append(comp_name)
            
            # Normalize energies
            total_energy = sum(energies)
            energies = [e/total_energy * 100 for e in energies]
            
            plt.bar([f'{col}_{name}' for name in comp_names], energies, alpha=0.7, label=col)
    
    plt.title('Energy Distribution Across DWT Components (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Components')
    plt.ylabel('Energy Percentage')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Frequency characteristics
    plt.subplot(3, 2, 6)
    
    # Compare the trend components (A5) of targets vs planets
    target_trends = []
    planet_trends = []
    
    for col in target_columns:
        if col in results:
            trend = results[col]['components'][f'A{num_components}']
            target_trends.append(np.std(trend))
    
    for col in planetary_columns:
        if col in results:
            trend = results[col]['components'][f'A{num_components}']
            planet_trends.append(np.std(trend))
    
    x_pos = np.arange(len(target_columns + planetary_columns))
    all_trends = target_trends + planet_trends
    all_names = target_columns + planetary_columns
    
    colors = ['red']*len(target_columns) + ['blue']*len(planetary_columns)
    plt.bar(x_pos, all_trends, alpha=0.7, color=colors)
    plt.title(f'Trend Component (A{num_components}) Variability', fontsize=14, fontweight='bold')
    plt.xlabel('Series')
    plt.ylabel('Standard Deviation')
    plt.xticks(x_pos, all_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/dwt_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_component_analysis(results: Dict, target_columns: List[str], 
                                     planetary_columns: List[str], num_components: int = 5):
    """Create detailed analysis of first 5 DWT components"""
    
    print(f"\n📊 Detailed Component Analysis (First {num_components} components):")
    print("=" * 70)
    
    # Create correlation matrix between components
    component_data = {}
    
    for col in target_columns + planetary_columns:
        if col in results:
            components = results[col]['components']
            
            # Store each component
            for comp_name, comp_signal in components.items():
                component_data[f'{col}_{comp_name}'] = comp_signal
    
    # Create DataFrame for correlation analysis
    min_length = min(len(signal) for signal in component_data.values())
    truncated_data = {name: signal[:min_length] for name, signal in component_data.items()}
    comp_df = pd.DataFrame(truncated_data)
    
    # Calculate correlation matrix
    correlation_matrix = comp_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('DWT Component Cross-Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/dwt_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Component statistics
    print(f"\n📈 Component Statistics Summary:")
    for col in target_columns + planetary_columns:
        if col in results:
            print(f"\n{col}:")
            components = results[col]['components']
            for comp_name, comp_signal in components.items():
                mean_val = np.mean(comp_signal)
                std_val = np.std(comp_signal)
                max_val = np.max(comp_signal)
                min_val = np.min(comp_signal)
                print(f"  {comp_name}: Mean={mean_val:.4f}, Std={std_val:.4f}, Range=[{min_val:.4f}, {max_val:.4f}]")

def save_dwt_results(results: Dict, output_path: str):
    """Save DWT decomposition results to CSV"""
    
    print(f"\n💾 Saving DWT results to: {output_path}")
    
    # Prepare data for saving
    all_data = {}
    
    for series_name, series_data in results.items():
        # Original signal
        all_data[f'{series_name}_original'] = series_data['signal']
        
        # DWT components
        for comp_name, comp_signal in series_data['components'].items():
            all_data[f'{series_name}_{comp_name}'] = comp_signal
    
    # Find minimum length to ensure all series have same length
    min_length = min(len(signal) for signal in all_data.values())
    
    # Truncate all series to minimum length
    truncated_data = {name: signal[:min_length] for name, signal in all_data.items()}
    
    # Create DataFrame and save
    dwt_df = pd.DataFrame(truncated_data)
    dwt_df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved {dwt_df.shape[0]} rows and {dwt_df.shape[1]} columns")
    print(f"   📊 Columns include original signals and DWT components")

def main():
    print("🌊 DWT Financial Analysis - Enhanced Version")
    print("=" * 50)
    
    # Load data
    df = load_financial_data('data/prepared_financial_data.csv')
    
    # Define columns to analyze
    target_columns = ['log_Open', 'log_High', 'log_Low', 'log_Close']  # First 4 non-date columns
    planetary_columns = ['Sun_sin', 'Saturn_sin', 'Jupiter_sin', 'Venus_sin']  # Planetary sine waves
    
    print(f"\n🎯 Analysis Configuration:")
    print(f"   Target columns: {target_columns}")
    print(f"   Planetary columns: {planetary_columns}")
    print(f"   DWT levels: 5")
    print(f"   Wavelet: db4 (Daubechies 4)")
    
    # Verify columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    missing_planets = [col for col in planetary_columns if col not in df.columns]
    
    if missing_targets:
        print(f"   ❌ Missing target columns: {missing_targets}")
    if missing_planets:
        print(f"   ❌ Missing planetary columns: {missing_planets}")
    
    # Filter to existing columns
    target_columns = [col for col in target_columns if col in df.columns]
    planetary_columns = [col for col in planetary_columns if col in df.columns]
    
    # Perform DWT analysis
    results = analyze_dwt_components(df, target_columns, planetary_columns, num_components=5)
    
    # Create visualizations
    create_dwt_comparison_plots(results, target_columns, planetary_columns, num_components=5)
    create_detailed_component_analysis(results, target_columns, planetary_columns, num_components=5)
    
    # Save results
    save_dwt_results(results, 'analysis/dwt_decomposition_results.csv')
    
    # Summary statistics
    print(f"\n📋 Analysis Summary:")
    print("=" * 40)
    total_components = None
    for series_name, series_data in results.items():
        if total_components is None:
            total_components = series_data['total_components']
        print(f"   {series_name}: {series_data['total_components']} total components")
    
    print(f"\n🔍 DWT Decomposition Structure:")
    print(f"   📊 Total levels: 5")
    print(f"   📈 Components per series: {total_components}")
    print(f"   🎯 Approximation: A5 (trend/low frequency)")
    print(f"   📊 Details: D5, D4, D3, D2, D1 (high to low frequency)")
    print(f"   💾 Saved results include all components for further analysis")
    
    print(f"\n✨ Analysis complete! Check the generated plots and CSV file.")

if __name__ == "__main__":
    main()

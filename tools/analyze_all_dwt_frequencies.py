#!/usr/bin/env python3
"""
Comprehensive DWT Frequency Analysis with Energy Distribution
Shows all 11 periods in one plot and calculates energy coverage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def analyze_all_frequencies_and_energy():
    """Analyze all DWT frequencies and their energy distribution"""
    
    # Load data
    print("📊 Loading prepared financial data...")
    df = pd.read_csv('data/prepared_financial_data.csv')
    
    # Use log_Close as our main target
    data = df['log_Close'].values
    print(f"✅ Loaded {len(data)} data points")
    
    # Apply DWT with maximum levels
    wavelet = 'db4'
    max_levels = pywt.dwt_max_level(len(data), wavelet)
    print(f"📈 Maximum DWT levels possible: {max_levels}")
    
    # Perform multi-level DWT
    coeffs = pywt.wavedec(data, wavelet, level=max_levels)
    
    # Calculate periods and energies
    sampling_period = 1  # 1 day
    periods = []
    energies = []
    energy_percentages = []
    component_names = []
    
    # Total energy (variance of original signal)
    total_energy = np.var(data)
    print(f"🔋 Total signal energy: {total_energy:.6f}")
    
    # Approximation component (lowest frequency)
    approx_energy = np.var(coeffs[0])
    approx_period = len(data) / (2**0)  # Approximation covers full length
    periods.append(approx_period)
    energies.append(approx_energy)
    energy_percentages.append((approx_energy / total_energy) * 100)
    component_names.append('A11 (Trend)')
    
    # Detail components (highest to lowest frequency)
    for i, coeff in enumerate(coeffs[1:], 1):
        # Period calculation: each level represents frequency bands
        period = sampling_period * (2 ** i)
        energy = np.var(coeff)
        energy_pct = (energy / total_energy) * 100
        
        periods.append(period)
        energies.append(energy)
        energy_percentages.append(energy_pct)
        component_names.append(f'D{i} ({period:.1f}d)')
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DWT Frequency Analysis: All 11 Components of log_Close', fontsize=16, fontweight='bold')
    
    # Plot 1: Energy vs Period
    colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
    bars1 = ax1.bar(range(len(periods)), energies, color=colors, alpha=0.7)
    ax1.set_xlabel('DWT Component')
    ax1.set_ylabel('Energy (Variance)')
    ax1.set_title('Energy Distribution Across Frequencies')
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(component_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add energy values on bars
    for i, (bar, energy) in enumerate(zip(bars1, energies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.01,
                f'{energy:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Energy Percentage
    wedges, texts, autotexts = ax2.pie(energy_percentages, labels=component_names, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Energy Percentage Distribution')
    
    # Plot 3: Period vs Energy (Log scale)
    ax3.loglog(periods, energies, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax3.set_xlabel('Period (Days)')
    ax3.set_ylabel('Energy (Variance)')
    ax3.set_title('Energy vs Period (Log-Log Scale)')
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add period labels
    for i, (period, energy) in enumerate(zip(periods, energies)):
        if period <= 20:  # Highlight sub-20 day periods
            ax3.annotate(f'{period:.1f}d', (period, energy), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red', fontweight='bold')
        else:
            ax3.annotate(f'{period:.1f}d', (period, energy), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='blue')
    
    # Plot 4: Cumulative Energy Coverage
    cumulative_energy = np.cumsum(energy_percentages)
    ax4.plot(range(len(cumulative_energy)), cumulative_energy, 'o-', 
             linewidth=3, markersize=8, color='green')
    ax4.set_xlabel('Number of Components')
    ax4.set_ylabel('Cumulative Energy Coverage (%)')
    ax4.set_title('Cumulative Energy Coverage')
    ax4.set_xticks(range(len(component_names)))
    ax4.set_xticklabels([f'{i+1}' for i in range(len(component_names))])
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax4.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax4.axhline(y=99, color='purple', linestyle='--', alpha=0.7, label='99% threshold')
    ax4.legend()
    
    # Add cumulative percentages as text
    for i, cum_energy in enumerate(cumulative_energy):
        ax4.text(i, cum_energy + 1, f'{cum_energy:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('analysis/all_11_dwt_components_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed analysis table
    analysis_df = pd.DataFrame({
        'Component': component_names,
        'Period_Days': periods,
        'Energy': energies,
        'Energy_Percentage': energy_percentages,
        'Cumulative_Energy': cumulative_energy
    })
    
    # Identify sub-20 day components
    sub20_mask = np.array(periods) <= 20
    sub20_components = analysis_df[sub20_mask]
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE DWT FREQUENCY ANALYSIS")
    print("="*60)
    print(f"Total Components: {len(component_names)}")
    print(f"Total Energy: {total_energy:.6f}")
    print(f"Energy Coverage by all components: {sum(energy_percentages):.1f}%")
    
    print(f"\n🔍 SUB-20 DAY COMPONENTS (Potential Noise?):")
    print("-"*50)
    for _, row in sub20_components.iterrows():
        print(f"{row['Component']:<12} | {row['Period_Days']:>8.1f} days | {row['Energy_Percentage']:>6.2f}% energy")
    
    sub20_total_energy = sub20_components['Energy_Percentage'].sum()
    print(f"\nTotal energy in sub-20 day components: {sub20_total_energy:.2f}%")
    
    print(f"\n📈 LONG-TERM COMPONENTS (>20 days):")
    print("-"*50)
    longterm_components = analysis_df[~sub20_mask]
    for _, row in longterm_components.iterrows():
        print(f"{row['Component']:<12} | {row['Period_Days']:>8.1f} days | {row['Energy_Percentage']:>6.2f}% energy")
    
    longterm_total_energy = longterm_components['Energy_Percentage'].sum()
    print(f"\nTotal energy in >20 day components: {longterm_total_energy:.2f}%")
    
    print(f"\n🎯 ENERGY COVERAGE MILESTONES:")
    print("-"*40)
    for threshold in [50, 80, 90, 95, 99]:
        components_needed = (cumulative_energy >= threshold).argmax() + 1
        print(f"{threshold}% coverage: First {components_needed} components")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("-"*30)
    
    # Find dominant frequency
    max_energy_idx = np.argmax(energy_percentages)
    dominant_component = component_names[max_energy_idx]
    dominant_period = periods[max_energy_idx]
    dominant_energy = energy_percentages[max_energy_idx]
    
    print(f"• Dominant frequency: {dominant_component} ({dominant_period:.1f} days, {dominant_energy:.1f}% energy)")
    print(f"• Sub-20 day energy (noise?): {sub20_total_energy:.1f}%")
    print(f"• Long-term structure: {longterm_total_energy:.1f}%")
    
    # Noise assessment
    if sub20_total_energy < 30:
        noise_assessment = "LOW - Most energy in meaningful frequencies"
    elif sub20_total_energy < 50:
        noise_assessment = "MODERATE - Significant but not dominant"
    else:
        noise_assessment = "HIGH - Consider filtering"
    
    print(f"• Noise level assessment: {noise_assessment}")
    
    # Save detailed results
    analysis_df.to_csv('analysis/dwt_frequency_energy_analysis.csv', index=False)
    print(f"\n💾 Saved detailed analysis to: analysis/dwt_frequency_energy_analysis.csv")
    
    return analysis_df, sub20_total_energy, longterm_total_energy

if __name__ == "__main__":
    analysis_results = analyze_all_frequencies_and_energy()

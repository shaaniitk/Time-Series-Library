#!/usr/bin/env python3
"""
Frequency Analysis: Signal vs Noise Detection for Financial Time Series

This script analyzes the frequency content of log_Close to determine:
1. How many meaningful frequencies exist below 20 days
2. What constitutes signal vs noise in high-frequency components
3. DWT component analysis with frequency mapping
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the financial data"""
    print("📊 Loading prepared financial data...")
    df = pd.read_csv('data/prepared_financial_data.csv')
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate sampling frequency (assuming daily data)
    time_diff = df.index[1] - df.index[0]
    if hasattr(time_diff, 'days'):
        sampling_freq = 1.0 / time_diff.days  # cycles per day
    else:
        sampling_freq = 1.0  # assume daily if unclear
    
    print(f"Sampling frequency: {sampling_freq} cycles/day")
    
    return df, sampling_freq

def frequency_to_period_days(freq, sampling_freq):
    """Convert frequency to period in days"""
    if freq == 0:
        return np.inf
    return 1.0 / (freq * sampling_freq)

def dwt_frequency_analysis(signal_data, sampling_freq, wavelet='db4', max_levels=10):
    """
    Perform DWT analysis and map components to frequency ranges
    """
    print(f"\\n🌊 DWT Frequency Analysis using {wavelet} wavelet")
    
    # Perform DWT decomposition
    coeffs = pywt.wavedec(signal_data, wavelet, level=max_levels)
    
    # Calculate frequency ranges for each level
    results = []
    
    for i, coeff in enumerate(coeffs):
        std_dev = np.std(coeff)
        variance = np.var(coeff)
        
        if i == 0:
            # Approximation coefficients (lowest frequencies)
            level_name = f"A{max_levels}"
            freq_range = f"DC - {sampling_freq/(2**(max_levels+1)):.4f}"
            period_range = f"> {2**(max_levels+1)} days"
        else:
            # Detail coefficients
            level = max_levels - i + 1
            level_name = f"D{level}"
            
            # Frequency range for this detail level
            freq_low = sampling_freq / (2**(level+1))
            freq_high = sampling_freq / (2**level)
            freq_range = f"{freq_low:.4f} - {freq_high:.4f}"
            
            # Period range in days
            period_high = 2**(level+1)
            period_low = 2**level
            period_range = f"{period_low} - {period_high} days"
        
        results.append({
            'Level': level_name,
            'Coefficients': len(coeff),
            'Std_Dev': std_dev,
            'Variance': variance,
            'Frequency_Range': freq_range,
            'Period_Range': period_range,
            'Energy': np.sum(coeff**2)
        })
    
    return results, coeffs

def fft_analysis(signal_data, sampling_freq):
    """
    Perform FFT analysis to identify dominant frequencies
    """
    print("\\n📈 FFT Frequency Analysis")
    
    # Remove DC component and detrend
    signal_detrended = signal.detrend(signal_data)
    
    # Apply window to reduce spectral leakage
    windowed_signal = signal_detrended * np.hanning(len(signal_detrended))
    
    # Compute FFT
    fft_vals = fft(windowed_signal)
    freqs = fftfreq(len(windowed_signal), d=1/sampling_freq)
    
    # Get positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power_pos = np.abs(fft_vals[pos_mask])**2
    
    # Convert to periods in days
    periods = 1.0 / freqs_pos
    
    return freqs_pos, power_pos, periods

def identify_signal_vs_noise(dwt_results, threshold_percentile=95):
    """
    Identify which frequency components are signal vs noise
    """
    print(f"\\n🔍 Signal vs Noise Analysis (>{threshold_percentile}th percentile = signal)")
    
    # Calculate energy threshold
    energies = [r['Energy'] for r in dwt_results]
    energy_threshold = np.percentile(energies, threshold_percentile)
    
    # Classify components
    signal_components = []
    noise_components = []
    
    for result in dwt_results:
        result['Signal_Type'] = 'Signal' if result['Energy'] > energy_threshold else 'Noise'
        
        if result['Signal_Type'] == 'Signal':
            signal_components.append(result)
        else:
            noise_components.append(result)
    
    return signal_components, noise_components, energy_threshold

def analyze_sub_20_day_frequencies(dwt_results, fft_freqs, fft_power, fft_periods):
    """
    Specifically analyze frequencies corresponding to periods < 20 days
    """
    print("\\n⚡ Sub-20 Day Frequency Analysis")
    
    # Filter FFT results for periods < 20 days
    sub_20_mask = fft_periods < 20
    sub_20_freqs = fft_freqs[sub_20_mask]
    sub_20_power = fft_power[sub_20_mask]
    sub_20_periods = fft_periods[sub_20_mask]
    
    if len(sub_20_freqs) == 0:
        print("No frequencies found with periods < 20 days")
        return
    
    # Sort by power (highest first)
    sorted_indices = np.argsort(sub_20_power)[::-1]
    
    print(f"Found {len(sub_20_freqs)} frequency components < 20 days:")
    print("Top 10 sub-20 day frequencies by power:")
    print("Rank | Period (days) | Frequency | Power | Power %")
    print("-" * 50)
    
    total_power = np.sum(fft_power)
    
    for i, idx in enumerate(sorted_indices[:10]):
        period = sub_20_periods[idx]
        freq = sub_20_freqs[idx]
        power = sub_20_power[idx]
        power_pct = (power / total_power) * 100
        
        print(f"{i+1:4d} | {period:11.2f} | {freq:9.4f} | {power:9.2e} | {power_pct:6.2f}%")
    
    # Analyze DWT components for sub-20 day periods
    print("\\nDWT Components with periods < 20 days:")
    sub_20_dwt = []
    for result in dwt_results:
        period_str = result['Period_Range']
        if 'days' in period_str and '-' in period_str:
            try:
                # Extract period range
                period_parts = period_str.replace(' days', '').split(' - ')
                if len(period_parts) == 2:
                    period_low = float(period_parts[0])
                    if period_low < 20:
                        sub_20_dwt.append(result)
            except:
                continue
    
    if sub_20_dwt:
        print("Level | Period Range | Energy | Signal/Noise")
        print("-" * 45)
        for result in sub_20_dwt:
            print(f"{result['Level']:5s} | {result['Period_Range']:12s} | {result['Energy']:6.2e} | {result['Signal_Type']}")
    
    return sub_20_freqs, sub_20_power, sub_20_periods

def create_visualizations(signal_data, dwt_results, coeffs, fft_freqs, fft_power, fft_periods):
    """
    Create comprehensive visualizations
    """
    print("\\n📊 Creating visualizations...")
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Original signal
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(signal_data, 'b-', linewidth=1)
    plt.title('Original log_Close Signal')
    plt.xlabel('Time (days)')
    plt.ylabel('log_Close')
    plt.grid(True, alpha=0.3)
    
    # 2. DWT Energy by Level
    ax2 = plt.subplot(3, 3, 2)
    levels = [r['Level'] for r in dwt_results]
    energies = [r['Energy'] for r in dwt_results]
    colors = ['red' if r['Signal_Type'] == 'Signal' else 'blue' for r in dwt_results]
    
    bars = plt.bar(levels, energies, color=colors, alpha=0.7)
    plt.title('DWT Energy by Level\\n(Red=Signal, Blue=Noise)')
    plt.xlabel('DWT Level')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. FFT Power Spectrum
    ax3 = plt.subplot(3, 3, 3)
    plt.loglog(fft_periods, fft_power, 'g-', linewidth=1)
    plt.axvline(x=20, color='red', linestyle='--', label='20 days')
    plt.title('FFT Power Spectrum')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. DWT Coefficients for first few levels
    for i, (level, coeff) in enumerate(zip(levels[:6], coeffs[:6])):
        ax = plt.subplot(3, 3, 4 + i)
        plt.plot(coeff, linewidth=1)
        plt.title(f'DWT {level} Coefficients')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/frequency_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved: analysis/frequency_analysis_comprehensive.png")
    
    # Zoomed FFT for sub-20 day periods
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Filter for periods 1-50 days for better visualization
    mask_zoom = (fft_periods >= 1) & (fft_periods <= 50)
    
    plt.semilogy(fft_periods[mask_zoom], fft_power[mask_zoom], 'b-', linewidth=1.5)
    plt.axvline(x=20, color='red', linestyle='--', linewidth=2, label='20 days threshold')
    plt.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='Weekly cycle')
    plt.axvline(x=30, color='green', linestyle='--', alpha=0.7, label='Monthly cycle')
    
    plt.title('FFT Power Spectrum: Focus on 1-50 Day Periods')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 50)
    
    plt.tight_layout()
    plt.savefig('analysis/sub_20_day_frequencies.png', dpi=300, bbox_inches='tight')
    print("Saved: analysis/sub_20_day_frequencies.png")

def main():
    """Main analysis function"""
    
    # Create analysis directory
    import os
    os.makedirs('analysis', exist_ok=True)
    
    print("🔬 Financial Time Series Frequency Analysis")
    print("=" * 50)
    
    # Load data
    df, sampling_freq = load_and_prepare_data()
    
    # Extract log_Close signal
    log_close = df['log_Close'].values
    
    print(f"\\nAnalyzing log_Close with {len(log_close)} data points")
    
    # DWT Analysis
    dwt_results, coeffs = dwt_frequency_analysis(log_close, sampling_freq)
    
    # FFT Analysis
    fft_freqs, fft_power, fft_periods = fft_analysis(log_close, sampling_freq)
    
    # Signal vs Noise classification
    signal_components, noise_components, energy_threshold = identify_signal_vs_noise(dwt_results)
    
    print(f"\\nEnergy threshold (95th percentile): {energy_threshold:.2e}")
    print(f"Signal components: {len(signal_components)}")
    print(f"Noise components: {len(noise_components)}")
    
    # Sub-20 day analysis
    sub_20_results = analyze_sub_20_day_frequencies(dwt_results, fft_freqs, fft_power, fft_periods)
    
    # Create visualizations
    create_visualizations(log_close, dwt_results, coeffs, fft_freqs, fft_power, fft_periods)
    
    # Save detailed results
    df_results = pd.DataFrame(dwt_results)
    df_results.to_csv('analysis/dwt_frequency_analysis.csv', index=False)
    print("\\nSaved: analysis/dwt_frequency_analysis.csv")
    
    # Summary
    print("\\n" + "="*50)
    print("📋 FREQUENCY ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\\n🎯 Key Findings:")
    print(f"• Total DWT levels analyzed: {len(dwt_results)}")
    print(f"• Signal components (>95th percentile energy): {len(signal_components)}")
    print(f"• Noise components: {len(noise_components)}")
    
    # Count sub-20 day signal components
    sub_20_signal = [r for r in dwt_results if r['Signal_Type'] == 'Signal' and '20' not in r['Period_Range'].split('-')[0]]
    print(f"• Sub-20 day signal components: {len(sub_20_signal)}")
    
    if len(signal_components) > 0:
        print(f"\\n🔊 Signal Components:")
        for comp in signal_components:
            print(f"  - {comp['Level']}: {comp['Period_Range']} (Energy: {comp['Energy']:.2e})")
    
    print(f"\\n🔇 Recommendation:")
    if len(sub_20_signal) > len(signal_components) * 0.5:
        print("  ⚠️  Many sub-20 day components contain signal - consider keeping")
    else:
        print("  ✅ Most sub-20 day components appear to be noise - safe to filter")

if __name__ == "__main__":
    main()

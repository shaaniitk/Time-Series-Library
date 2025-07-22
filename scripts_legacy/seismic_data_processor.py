import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert, find_peaks
import warnings
warnings.filterwarnings('ignore')

class SeismicDataProcessor:
    """Process multi-wave seismic data with phase and velocity corrections"""
    
    def __init__(self):
        self.scalers = {}
        self.phase_correctors = {}
    
    def extract_wave_features(self, wave_data, sampling_rate=100):
        """Extract key features from wave data"""
        # Instantaneous amplitude and phase using Hilbert transform
        analytic_signal = hilbert(wave_data)
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)
        
        # Dominant frequency
        fft = np.fft.fft(wave_data)
        freqs = np.fft.fftfreq(len(wave_data), 1/sampling_rate)
        dominant_freq = freqs[np.argmax(np.abs(fft))]
        
        # Peak detection for extreme events
        peaks, _ = find_peaks(amplitude, height=np.std(amplitude)*2)
        peak_intensity = np.mean(amplitude[peaks]) if len(peaks) > 0 else 0
        
        return {
            'amplitude': amplitude,
            'phase': phase,
            'dominant_freq': dominant_freq,
            'peak_intensity': peak_intensity
        }
    
    def align_phases(self, waves_dict):
        """Align multiple waves by phase correction"""
        reference_wave = list(waves_dict.keys())[0]
        ref_phase = waves_dict[reference_wave]['phase']
        
        aligned_waves = {}
        for wave_name, wave_data in waves_dict.items():
            if wave_name == reference_wave:
                aligned_waves[wave_name] = wave_data
            else:
                # Simple phase alignment - can be made more sophisticated
                phase_diff = np.mean(wave_data['phase'] - ref_phase)
                aligned_amplitude = wave_data['amplitude'] * np.cos(phase_diff)
                aligned_waves[wave_name] = {
                    **wave_data,
                    'amplitude': aligned_amplitude
                }
        
        return aligned_waves
    
    def create_lagged_features(self, df, lag_steps=[1, 6, 12, 24]):
        """Create time-lagged features for complex interactions"""
        lagged_df = df.copy()
        
        for col in df.columns:
            if col != 'timestamp':
                for lag in lag_steps:
                    lagged_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return lagged_df.dropna()
    
    def detect_extreme_events(self, seismic_data, threshold_std=3):
        """Mark extreme seismic events for focused training"""
        mean_val = np.mean(seismic_data)
        std_val = np.std(seismic_data)
        
        extreme_mask = np.abs(seismic_data - mean_val) > threshold_std * std_val
        return extreme_mask
    
    def prepare_training_data(self, covariate_waves, seismic_target, 
                            seq_len=168, pred_len=24):
        """Prepare data for TSLib training"""
        
        # Process each covariate wave
        processed_waves = {}
        for wave_name, wave_data in covariate_waves.items():
            features = self.extract_wave_features(wave_data)
            processed_waves[wave_name] = features
        
        # Align phases
        aligned_waves = self.align_phases(processed_waves)
        
        # Create DataFrame
        data_dict = {'timestamp': pd.date_range('2020-01-01', 
                                               periods=len(seismic_target), 
                                               freq='H')}
        
        # Add wave features
        for wave_name, wave_features in aligned_waves.items():
            data_dict[f'{wave_name}_amplitude'] = wave_features['amplitude']
            data_dict[f'{wave_name}_phase'] = wave_features['phase']
            data_dict[f'{wave_name}_freq'] = [wave_features['dominant_freq']] * len(seismic_target)
        
        # Add target
        data_dict['seismic_amplitude'] = seismic_target
        
        df = pd.DataFrame(data_dict)
        
        # Create lagged features for complex interactions
        df_lagged = self.create_lagged_features(df)
        
        # Mark extreme events
        extreme_mask = self.detect_extreme_events(seismic_target)
        df_lagged['is_extreme'] = extreme_mask[:len(df_lagged)]
        
        return df_lagged

# Example usage function
def create_sample_seismic_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate covariate waves with different properties
    t = np.linspace(0, 100, n_samples)
    
    covariate_waves = {
        'p_wave': np.sin(2*np.pi*0.1*t) + 0.1*np.random.randn(n_samples),
        's_wave': np.sin(2*np.pi*0.05*t + np.pi/4) + 0.1*np.random.randn(n_samples),
        'surface_wave': np.sin(2*np.pi*0.02*t + np.pi/2) + 0.1*np.random.randn(n_samples)
    }
    
    # Simulate complex seismic response
    seismic_target = (0.5 * covariate_waves['p_wave'] + 
                     0.3 * np.roll(covariate_waves['s_wave'], 10) +  # Time lag
                     0.2 * np.roll(covariate_waves['surface_wave'], 20) +
                     0.1 * np.random.randn(n_samples))
    
    # Add some extreme events
    extreme_indices = np.random.choice(n_samples, 20, replace=False)
    seismic_target[extreme_indices] += np.random.randn(20) * 3
    
    return covariate_waves, seismic_target

if __name__ == "__main__":
    # Test the processor
    processor = SeismicDataProcessor()
    covariate_waves, seismic_target = create_sample_seismic_data()
    
    # Process data
    processed_df = processor.prepare_training_data(covariate_waves, seismic_target)
    
    # Save for TSLib
    processed_df.to_csv('./dataset/seismic/seismic_waves.csv', index=False)
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")
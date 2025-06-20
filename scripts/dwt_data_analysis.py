#!/usr/bin/env python3
"""
Data Analysis with Discrete Wavelet Transform (DWT)

This script loads financial time series data, applies DWT decomposition,
saves the transformed data, and provides comprehensive visualizations
to understand the nature of the data.

Features:
- Load and analyze CSV data
- Apply DWT with multiple wavelets
- Decompose into trend, seasonal, and noise components
- Generate comprehensive plots
- Save DWT-transformed data
- Provide data insights and statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DWTAnalyzer:
    """Comprehensive DWT analysis for time series data"""
    
    def __init__(self, data_path: str, date_column: str = 'date'):
        """
        Initialize DWT analyzer
        
        Args:
            data_path: Path to CSV file
            date_column: Name of the date column
        """
        self.data_path = data_path
        self.date_column = date_column
        self.df = None
        self.dwt_results = {}
        self.wavelets = ['db4', 'db8', 'haar', 'coif2', 'bior2.2']
        
        print(f"🔍 DWT Analyzer initialized for: {data_path}")
    
    def load_data(self):
        """Load and prepare data"""
        print("📊 Loading data...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")
        
        # Convert date column
        if self.date_column in self.df.columns:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            self.df.set_index(self.date_column, inplace=True)
        
        # Get numeric columns
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   Numeric columns: {len(self.numeric_columns)}")
        
        # Basic statistics
        print(f"\n📈 Data Overview:")
        print(f"   Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"   Missing values: {self.df.isnull().sum().sum()}")
        print(f"   Data types: {self.df.dtypes.value_counts().to_dict()}")
        
        return self.df
    
    def apply_dwt(self, column: str, wavelet: str = 'db4', levels: int = 4):
        """
        Apply DWT to a specific column
        
        Args:
            column: Column name to analyze
            wavelet: Wavelet type
            levels: Decomposition levels
            
        Returns:
            Dictionary with DWT coefficients
        """
        data = self.df[column].dropna().values
        
        # Ensure data length is appropriate for decomposition
        max_levels = pywt.dwt_max_level(len(data), wavelet)
        levels = min(levels, max_levels)
        
        # Perform multi-level DWT
        coeffs = pywt.wavedec(data, wavelet, level=levels)
        
        # Reconstruct components
        reconstructed = []
        for i in range(len(coeffs)):
            # Create zero coefficients except for level i
            temp_coeffs = [np.zeros_like(c) for c in coeffs]
            temp_coeffs[i] = coeffs[i]
            reconstructed.append(pywt.waverec(temp_coeffs, wavelet))
        
        result = {
            'original': data,
            'coefficients': coeffs,
            'reconstructed': reconstructed,
            'levels': levels,
            'wavelet': wavelet,
            'approximation': reconstructed[0],  # Low frequency (trend)
            'details': reconstructed[1:],        # High frequency (details)
            'energy': [np.sum(c**2) for c in coeffs]
        }
        
        return result
    
    def analyze_all_columns(self, max_columns: int = 10):
        """Apply DWT to all numeric columns"""
        print(f"\n🌊 Applying DWT to columns...")
        
        # Limit to most important columns if too many
        columns_to_analyze = self.numeric_columns[:max_columns]
        
        for column in columns_to_analyze:
            print(f"   Processing: {column}")
            
            # Try different wavelets
            self.dwt_results[column] = {}
            for wavelet in self.wavelets[:3]:  # Use top 3 wavelets
                try:
                    result = self.apply_dwt(column, wavelet)
                    self.dwt_results[column][wavelet] = result
                except Exception as e:
                    print(f"     Warning: {wavelet} failed for {column}: {e}")
        
        print(f"✅ DWT analysis completed for {len(self.dwt_results)} columns")
    
    def create_dwt_dataframe(self):
        """Create DataFrame with DWT components"""
        print("\n🔄 Creating DWT-transformed dataset...")
        
        dwt_data = {'date': self.df.index}
        
        for column in self.dwt_results.keys():
            # Use db4 results (most common)
            if 'db4' in self.dwt_results[column]:
                result = self.dwt_results[column]['db4']
                
                # Add original data
                dwt_data[f'{column}_original'] = result['original']
                
                # Add approximation (trend)
                dwt_data[f'{column}_trend'] = result['approximation']
                
                # Add detail levels
                for i, detail in enumerate(result['details']):
                    dwt_data[f'{column}_detail_{i+1}'] = detail
                
                # Add reconstructed signal
                reconstructed = result['approximation']
                for detail in result['details']:
                    reconstructed += detail
                dwt_data[f'{column}_reconstructed'] = reconstructed
        
        # Create DataFrame
        max_length = max(len(v) for v in dwt_data.values() if isinstance(v, np.ndarray))
        
        # Pad shorter arrays
        for key, value in dwt_data.items():
            if isinstance(value, np.ndarray) and len(value) < max_length:
                dwt_data[key] = np.pad(value, (0, max_length - len(value)), 'edge')
        
        # Handle date index
        if len(dwt_data['date']) < max_length:
            # Extend dates if needed
            last_date = dwt_data['date'][-1]
            freq = pd.infer_freq(dwt_data['date'][:10])
            if freq is None:
                freq = 'H'  # Default to hourly
            extended_dates = pd.date_range(start=last_date, periods=max_length-len(dwt_data['date'])+1, freq=freq)[1:]
            dwt_data['date'] = np.concatenate([dwt_data['date'], extended_dates])
        
        self.dwt_df = pd.DataFrame(dwt_data)
        self.dwt_df.set_index('date', inplace=True)
        
        print(f"   DWT DataFrame shape: {self.dwt_df.shape}")
        return self.dwt_df
    
    def plot_dwt_analysis(self, column: str, save_plots: bool = True):
        """Create comprehensive DWT plots for a column"""
        if column not in self.dwt_results:
            print(f"No DWT results for column: {column}")
            return
        
        # Use db4 results
        result = self.dwt_results[column]['db4']
        
        fig, axes = plt.subplots(2 + len(result['details']), 1, figsize=(15, 12))
        fig.suptitle(f'DWT Analysis: {column}', fontsize=16, fontweight='bold')
        
        # Original signal
        axes[0].plot(result['original'], 'b-', linewidth=1)
        axes[0].set_title('Original Signal')
        axes[0].grid(True, alpha=0.3)
        
        # Approximation (trend)
        axes[1].plot(result['approximation'], 'r-', linewidth=2)
        axes[1].set_title('Approximation (Trend Component)')
        axes[1].grid(True, alpha=0.3)
        
        # Detail levels
        for i, detail in enumerate(result['details']):
            axes[2 + i].plot(detail, 'g-', linewidth=1)
            axes[2 + i].set_title(f'Detail Level {i+1} (High Frequency)')
            axes[2 + i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'dwt_analysis_{column}.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Saved plot: dwt_analysis_{column}.png")
        
        plt.show()
    
    def plot_energy_distribution(self, save_plots: bool = True):
        """Plot energy distribution across DWT levels"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DWT Energy Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Energy by level for each column
        energy_data = []
        for column in self.dwt_results.keys():
            if 'db4' in self.dwt_results[column]:
                energy = self.dwt_results[column]['db4']['energy']
                for level, eng in enumerate(energy):
                    energy_data.append({
                        'Column': column,
                        'Level': f'Level_{level}' if level > 0 else 'Approximation',
                        'Energy': eng,
                        'Energy_Percent': eng / sum(energy) * 100
                    })
        
        energy_df = pd.DataFrame(energy_data)
        
        # Energy by level
        sns.boxplot(data=energy_df, x='Level', y='Energy_Percent', ax=axes[0,0])
        axes[0,0].set_title('Energy Distribution by DWT Level')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Energy by column
        sns.barplot(data=energy_df[energy_df['Level'] == 'Approximation'], 
                   x='Column', y='Energy_Percent', ax=axes[0,1])
        axes[0,1].set_title('Trend Energy by Column')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Wavelet comparison for first column
        first_col = list(self.dwt_results.keys())[0]
        wavelet_energy = []
        for wavelet in self.dwt_results[first_col].keys():
            total_energy = sum(self.dwt_results[first_col][wavelet]['energy'])
            wavelet_energy.append({'Wavelet': wavelet, 'Total_Energy': total_energy})
        
        wavelet_df = pd.DataFrame(wavelet_energy)
        sns.barplot(data=wavelet_df, x='Wavelet', y='Total_Energy', ax=axes[1,0])
        axes[1,0].set_title(f'Wavelet Performance Comparison ({first_col})')
        
        # Correlation matrix of trend components
        trend_columns = [col for col in self.dwt_df.columns if '_trend' in col]
        if len(trend_columns) > 1:
            corr_matrix = self.dwt_df[trend_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('Trend Components Correlation')
        else:
            axes[1,1].text(0.5, 0.5, 'Need multiple columns\nfor correlation', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Trend Correlation (N/A)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('dwt_energy_analysis.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Saved plot: dwt_energy_analysis.png")
        
        plt.show()
    
    def plot_data_overview(self, save_plots: bool = True):
        """Create overview plots of the original data"""
        # Select top 6 columns for visualization
        viz_columns = self.numeric_columns[:6]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Original Data Overview', fontsize=16, fontweight='bold')
        
        for i, column in enumerate(viz_columns):
            if i >= len(axes):
                break
                
            # Time series plot
            self.df[column].plot(ax=axes[i], title=f'{column} - Time Series')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {self.df[column].mean():.3f}\n'
            stats_text += f'Std: {self.df[column].std():.3f}\n'
            stats_text += f'Skew: {stats.skew(self.df[column].dropna()):.3f}'
            
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(viz_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Saved plot: data_overview.png")
        
        plt.show()
    
    def generate_insights(self):
        """Generate data insights and recommendations"""
        print("\n🧠 Data Insights and Analysis")
        print("=" * 50)
        
        # Basic data characteristics
        print(f"📊 Dataset Characteristics:")
        print(f"   • Total samples: {len(self.df)}")
        print(f"   • Features: {len(self.numeric_columns)}")
        print(f"   • Time span: {(self.df.index[-1] - self.df.index[0]).days} days")
        print(f"   • Frequency: {pd.infer_freq(self.df.index[:10]) or 'Unknown'}")
        
        # Missing data analysis
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        if missing_pct.sum() > 0:
            print(f"\n⚠️  Missing Data:")
            for col, pct in missing_pct[missing_pct > 0].items():
                print(f"   • {col}: {pct:.1f}%")
        else:
            print(f"\n✅ No missing data found")
        
        # Volatility analysis
        print(f"\n📈 Volatility Analysis:")
        volatilities = self.df[self.numeric_columns].std().sort_values(ascending=False)
        print(f"   • Most volatile: {volatilities.index[0]} (σ={volatilities.iloc[0]:.3f})")
        print(f"   • Least volatile: {volatilities.index[-1]} (σ={volatilities.iloc[-1]:.3f})")
        
        # DWT insights
        if self.dwt_results:
            print(f"\n🌊 DWT Analysis Insights:")
            
            # Trend strength analysis
            trend_strengths = {}
            for column in self.dwt_results.keys():
                if 'db4' in self.dwt_results[column]:
                    energy = self.dwt_results[column]['db4']['energy']
                    trend_strength = energy[0] / sum(energy) * 100
                    trend_strengths[column] = trend_strength
            
            if trend_strengths:
                strongest_trend = max(trend_strengths, key=trend_strengths.get)
                weakest_trend = min(trend_strengths, key=trend_strengths.get)
                
                print(f"   • Strongest trend component: {strongest_trend} ({trend_strengths[strongest_trend]:.1f}%)")
                print(f"   • Weakest trend component: {weakest_trend} ({trend_strengths[weakest_trend]:.1f}%)")
                print(f"   • Average trend strength: {np.mean(list(trend_strengths.values())):.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        avg_trend_strength = np.mean(list(trend_strengths.values())) if trend_strengths else 0
        
        if avg_trend_strength > 70:
            print(f"   • Data is highly trend-dominated - good for trend-following models")
            print(f"   • Consider using trend components for prediction")
        elif avg_trend_strength > 40:
            print(f"   • Balanced trend/noise ratio - suitable for most forecasting models")
            print(f"   • DWT preprocessing may improve performance")
        else:
            print(f"   • High noise/volatility detected - consider denoising")
            print(f"   • Use robust models or ensemble methods")
        
        if len(self.numeric_columns) > 10:
            print(f"   • High-dimensional data - consider dimensionality reduction")
            print(f"   • Feature selection may improve model performance")
        
        print(f"   • Best wavelets for your data: {', '.join(self.wavelets[:3])}")
        print(f"   • Recommended decomposition levels: 4-6")
    
    def save_dwt_data(self, output_path: str = None):
        """Save DWT-transformed data"""
        if output_path is None:
            output_path = self.data_path.replace('.csv', '_dwt_transformed.csv')
        
        if hasattr(self, 'dwt_df'):
            self.dwt_df.to_csv(output_path)
            print(f"\n💾 DWT-transformed data saved to: {output_path}")
            print(f"   Shape: {self.dwt_df.shape}")
            print(f"   Columns: {len(self.dwt_df.columns)}")
        else:
            print("❌ No DWT data to save. Run create_dwt_dataframe() first.")
        
        return output_path

def main():
    """Main analysis function"""
    print("🌊 Discrete Wavelet Transform (DWT) Data Analysis")
    print("=" * 60)
    
    # Configuration
    data_path = "data/comprehensive_dynamic_features.csv"  # Your main dataset
    
    # Initialize analyzer
    analyzer = DWTAnalyzer(data_path)
    
    # Load and analyze data
    analyzer.load_data()
    analyzer.plot_data_overview()
    
    # Apply DWT
    analyzer.analyze_all_columns(max_columns=8)  # Analyze top 8 columns
    
    # Create DWT dataset
    dwt_df = analyzer.create_dwt_dataframe()
    
    # Generate visualizations
    analyzer.plot_energy_distribution()
    
    # Plot detailed analysis for top 3 columns
    for column in list(analyzer.dwt_results.keys())[:3]:
        analyzer.plot_dwt_analysis(column)
    
    # Generate insights
    analyzer.generate_insights()
    
    # Save transformed data
    output_path = analyzer.save_dwt_data()
    
    print(f"\n✅ DWT Analysis Complete!")
    print(f"📊 Plots saved as PNG files")
    print(f"💾 Transformed data: {output_path}")
    print(f"🧠 Review the insights above for data characteristics")

if __name__ == "__main__":
    main()

import os
import sys
import torch
import numpy as np
from types import SimpleNamespace
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from seismic_data_processor import SeismicDataProcessor, create_sample_seismic_data

class SeismicWaveTrainer:
    """Custom trainer for seismic wave prediction"""
    
    def __init__(self, config_path='seismic_wave_config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.setup_data()
    
    def load_config(self):
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.args = SimpleNamespace(**config)
        
        # Set hardware
        if torch.cuda.is_available():
            self.args.use_gpu = True
            self.args.gpu_type = 'cuda'
        else:
            self.args.use_gpu = False
            self.args.gpu_type = 'cpu'
    
    def setup_data(self):
        """Setup seismic data"""
        # Create dataset directory
        os.makedirs('./dataset/seismic', exist_ok=True)
        
        # Process seismic data
        processor = SeismicDataProcessor()
        covariate_waves, seismic_target = create_sample_seismic_data()
        
        # Prepare training data
        processed_df = processor.prepare_training_data(covariate_waves, seismic_target)
        
        # Save processed data
        processed_df.to_csv('./dataset/seismic/seismic_waves.csv', index=False)
        
        print(f"Data prepared: {processed_df.shape}")
        print(f"Features: {processed_df.columns.tolist()}")
        
        # Update config with actual feature count
        self.args.enc_in = len([col for col in processed_df.columns 
                               if col not in ['timestamp', 'seismic_amplitude', 'is_extreme']])
        self.args.dec_in = self.args.enc_in
    
    def train_model(self):
        """Train the seismic prediction model"""
        print("Starting seismic wave prediction training...")
        
        # Initialize experiment
        exp = Exp_Long_Term_Forecast(self.args)
        
        # Create setting string
        setting = f"{self.args.model_id}_{self.args.features}_{self.args.seq_len}_{self.args.pred_len}"
        
        # Train model
        print("Training model...")
        exp.train(setting)
        
        # Test model
        print("Testing model...")
        exp.test(setting, test=1)
        
        print("Training completed!")
        return exp, setting
    
    def predict_extremes(self, exp, setting, threshold=2.0):
        """Specialized prediction for extreme events"""
        # Load test data
        test_data, test_loader = exp._get_data(flag='test')
        
        predictions = []
        actuals = []
        extreme_events = []
        
        exp.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                if exp.args.use_gpu:
                    batch_x = batch_x.float().cuda()
                    batch_y = batch_y.float().cuda()
                
                # Prediction
                outputs = exp.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                predictions.append(pred)
                actuals.append(true)
                
                # Detect extreme predictions
                extreme_mask = np.abs(pred) > threshold * np.std(pred)
                extreme_events.append(extreme_mask)
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        extreme_events = np.concatenate(extreme_events, axis=0)
        
        print(f"Predicted {np.sum(extreme_events)} extreme events")
        
        return predictions, actuals, extreme_events

def main():
    """Main training function"""
    # Initialize trainer
    trainer = SeismicWaveTrainer()
    
    # Train model
    exp, setting = trainer.train_model()
    
    # Predict extreme events
    predictions, actuals, extremes = trainer.predict_extremes(exp, setting)
    
    print(f"Training completed. Model saved as: {setting}")
    print(f"Extreme events detected: {np.sum(extremes)}")

if __name__ == "__main__":
    main()
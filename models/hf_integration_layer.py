"""
Integration Layer for Hugging Face Autoformers (Modularized)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from argparse import Namespace
from pathlib import Path

from .HuggingFaceAutoformerSuite import create_hf_autoformer, UncertaintyResult
from utils.logger import logger

class ModelMigrator:
    @staticmethod
    def analyze_custom_model_configs(custom_model_path: str) -> Dict[str, Any]:
        logger.info(f"Analyzing custom model: {custom_model_path}")
        try:
            checkpoint = torch.load(custom_model_path, map_location='cpu')
            model_state = checkpoint.get('model_state_dict', {})
            total_params = sum(p.numel() for p in model_state.values())
            
            size = "tiny"
            if total_params > 1e7: size = "large"
            elif total_params > 1e6: size = "base"
            elif total_params > 1e5: size = "small"

            return {'model_size_estimate': size, 'migration_complexity': 'low'}
        except Exception as e:
            logger.warning(f"Could not analyze custom model: {e}")
            return {'migration_complexity': 'unknown'}

class HFModelAdapter:
    def __init__(self, model_type: str, configs, backend: Optional[str] = None, migration_path: Optional[str] = None, **kwargs):
        self.model_type = model_type
        self.configs = configs
        
        migration_info = {}
        if migration_path and Path(migration_path).exists():
            migration_info = ModelMigrator.analyze_custom_model_configs(migration_path)
        
        backend = backend or migration_info.get('recommended_hf_backend', 'chronos')
        
        self.hf_model = create_hf_autoformer(model_type, configs, model_backend=backend, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.hf_model(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.hf_model.forward(*args, **kwargs)
    
    def configure_optimizer_loss(self, base_criterion, verbose=False):
        def hf_loss_wrapper(predictions, targets):
            if isinstance(predictions, UncertaintyResult):
                predictions = predictions.prediction
            loss = base_criterion(predictions, targets)
            if verbose:
                logger.info(f"HF Model Loss: {loss.item():.6f}")
            return loss
        return hf_loss_wrapper
    
    def get_kl_loss(self, max_kl_value=1e6):
        return 0.0
    
    def compute_loss(self, predictions, targets, base_criterion, return_components=False):
        if isinstance(predictions, UncertaintyResult):
            predictions = predictions.prediction
        loss = base_criterion(predictions, targets)
        if return_components:
            return {'data_loss': loss, 'kl_contribution': torch.tensor(0.0), 'total_loss': loss}
        return loss

def migrate_bayesian_autoformer(configs, existing_model_path: Optional[str] = None, **kwargs):
    return HFModelAdapter("BayesianEnhancedAutoformer", configs, migration_path=existing_model_path, **kwargs)

def migrate_hierarchical_autoformer(configs, existing_model_path: Optional[str] = None, **kwargs):
    return HFModelAdapter("HierarchicalEnhancedAutoformer", configs, migration_path=existing_model_path, **kwargs)

def migrate_quantile_autoformer(configs, existing_model_path: Optional[str] = None, **kwargs):
    return HFModelAdapter("QuantileBayesianAutoformer", configs, migration_path=existing_model_path, **kwargs)
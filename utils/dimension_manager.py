import os
import pandas as pd
import yaml
from typing import Dict, Any, List, Optional, Union
from utils.logger import logger
# Assuming analyze_dataset is in utils/data_analysis.py
# You might need to adjust the import path based on your file structure
try:
    from utils.data_analysis import analyze_dataset
except ImportError:
    logger.error("Could not import analyze_dataset from utils.data_analysis. Please ensure the file exists and the import path is correct.")
    # Define a dummy function or raise an error if analyze_dataset is essential
    def analyze_dataset(*args, **kwargs):
        raise NotImplementedError("analyze_dataset function is required but could not be imported.")

from argparse import Namespace # Import Namespace

class DimensionManager:
    """
    Manages input, output, and internal dimensions based on data, config, task, and loss.
    Acts as a single source of truth for model and evaluation dimensions.
    """

    def __init__(self):
        self._analysis_results: Optional[Dict] = None
        self._resolved_dimensions: Dict[str, Any] = {}
        self._config: Optional[Dict] = None
        self._args: Optional[Any] = None # Store args Namespace if available
        logger.info("DimensionManager initialized.")

    def analyze_and_configure(self,
                              data_path: str,
                              config: Optional[Dict] = None,
                              args: Optional[Any] = None, # Pass args Namespace
                              task_name: str = 'long_term_forecast',
                              features_mode: str = 'MS',
                              target_columns: Optional[Union[str, List[str]]] = None,
                              loss_name: str = 'mse',
                              quantile_levels: Optional[List[float]] = None):
        """
        Analyzes the dataset and configures dimensions.

        Args:
            data_path: Path to the dataset file.
            config: Optional config dictionary.
            args: Optional argparse Namespace.
            task_name: The task name (e.g., 'long_term_forecast').
            features_mode: The features mode ('M', 'MS', 'S').
            target_columns: Explicit target columns (overrides config/args).
            loss_name: The name of the loss function.
            quantile_levels: List of quantile levels for quantile regression.
        """
        self._config = config
        self._args = args
        self._task_name = task_name
        self._features_mode = features_mode
        self._loss_name = loss_name
        self._quantile_levels = quantile_levels

        logger.info(f"Analyzing data and configuring dimensions for task='{task_name}', features='{features_mode}', loss='{loss_name}'")

        # 1. Analyze the dataset
        try:
            # Use analyze_dataset from data_analysis.py
            # Pass target_columns from args/config if not explicitly provided
            if target_columns is None and args is not None and hasattr(args, 'target'):
                 target_columns = args.target
            elif target_columns is None and config is not None and 'target' in config:
                 target_columns = config['target']

            # analyze_dataset might need the full path, construct it if root_path is in args/config
            full_data_path = data_path
            if args is not None and hasattr(args, 'root_path') and args.root_path and not os.path.isabs(data_path):
                 full_data_path = os.path.join(args.root_path, data_path)
            elif config is not None and 'root_path' in config and config['root_path'] and not os.path.isabs(data_path):
                 full_data_path = os.path.join(config['root_path'], data_path)

            self._analysis_results = analyze_dataset(full_data_path, target_columns=target_columns)
            logger.info("Dataset analysis complete.")
        except FileNotFoundError:
            logger.error(f"Dataset not found at {data_path}. Cannot analyze dimensions.")
            raise # Re-raise the error
        except Exception as e:
            logger.error(f"Error during dataset analysis: {e}")
            raise # Re-raise the error

        # 2. Resolve dimensions based on analysis, mode, task, and loss
        self._resolve_dimensions()

    def _resolve_dimensions(self):
        """Resolves and stores all relevant dimensions."""
        if self._analysis_results is None:
            raise RuntimeError("Dataset analysis must be performed before resolving dimensions.")

        analysis = self._analysis_results
        mode = self._features_mode
        task = self._task_name
        loss = self._loss_name
        q_levels = self._quantile_levels

        # Get base dimensions from data analysis based on mode
        mode_analysis = analysis.get(f'mode_{mode}')
        if mode_analysis is None:
             raise ValueError(f"Unsupported features mode '{mode}' found in data analysis results.")

        enc_in_base = mode_analysis.get('enc_in')
        dec_in_base = mode_analysis.get('dec_in')
        c_out_eval = mode_analysis.get('c_out') # This is the number of true target features for evaluation

        if enc_in_base is None or dec_in_base is None or c_out_eval is None:
             raise ValueError(f"Incomplete dimension analysis for mode '{mode}': {mode_analysis}")

        # Determine num_quantiles
        num_quantiles = 1
        if q_levels and isinstance(q_levels, list) and len(q_levels) > 0:
            num_quantiles = len(q_levels)
            logger.info(f"Quantile levels provided: {q_levels}. Num quantiles: {num_quantiles}")
        else:
             logger.info("No quantile levels provided. Assuming point prediction (num_quantiles=1).")

        # Determine the model's final output layer dimension (c_out_model)
        # This depends on the task and loss function requirements
        c_out_model = c_out_eval # Default to predicting the base target features

        if task == 'long_term_forecast' or task == 'short_term_forecast':
            if loss == 'pinball' or (loss == 'quantile' and num_quantiles > 1):
                 # For multi-quantile forecasting, the model's final layer outputs C_eval * Q features
                 c_out_model = c_out_eval * num_quantiles
                 logger.info(f"Task '{task}' with loss '{loss}' and {num_quantiles} quantiles. Model output dimension set to c_out_eval * num_quantiles = {c_out_eval} * {num_quantiles} = {c_out_model}")
            else:
                 # For standard forecasting losses, model outputs C_eval features
                 c_out_model = c_out_eval
                 logger.info(f"Task '{task}' with loss '{loss}'. Model output dimension set to c_out_eval = {c_out_model}")
        elif task == 'imputation' or task == 'anomaly_detection':
             # For imputation/AD, model typically outputs all input features (enc_in)
             # or sometimes just the original c_out if it's different.
             # Let's assume it outputs enc_in features for these tasks based on typical models.
             c_out_model = enc_in_base # Or should this be analysis['n_total_features']? Let's use enc_in_base for consistency with mode.
             logger.info(f"Task '{task}'. Model output dimension set to enc_in_base = {c_out_model}")
        elif task == 'classification':
             # Classification output is num_classes, not related to time series features
             # This manager is primarily for time series feature dimensions, classification is different.
             # We might need a separate manager or handle this task differently.
             # For now, let's set c_out_model to 1 as a placeholder, but this needs review for classification.
             c_out_model = 1 # Placeholder for classification output dimension
             logger.warning(f"Task '{task}' detected. Dimension management for classification output (num_classes) is not fully handled by this manager.")

        # Store resolved dimensions
        self._resolved_dimensions = {
            'enc_in': enc_in_base,
            'dec_in': dec_in_base,
            'c_out_model': c_out_model, # Dimension for the model's final output layer
            'c_out_evaluation': c_out_eval, # Number of base target features for metrics/scaling
            'num_quantiles': num_quantiles,
            'quantile_levels': q_levels,
            'n_total_features_data': analysis.get('n_total_features'), # Total features in the raw data file
            'n_targets_data': analysis.get('n_targets'), # Number of target columns in the raw data file
            'target_columns': analysis.get('target_columns'), # List of target column names
            'data_path': analysis.get('data_path'),
            'features_mode': mode,
            'task_name': task,
            'loss_name': loss
        }

        logger.info(f"Resolved dimensions: {self._resolved_dimensions}")

    def get_model_init_params(self) -> Namespace:
        """
        Returns a Namespace of parameters suitable for passing to model __init__.
        Combines original args with resolved dimensions.
        """
        if not self._resolved_dimensions:
            raise RuntimeError("Dimensions must be resolved before getting model init params.")

        if self._args is None:
             logger.warning("DimensionManager was not initialized with args Namespace. Returning only resolved dimensions.")
             # Create a minimal Namespace with just resolved dimensions
             return Namespace(**self._resolved_dimensions)

        # Create a copy of the original args Namespace
        args_dict = vars(self._args).copy()

        # Override dimension-related args with resolved values
        args_dict['enc_in'] = self._resolved_dimensions['enc_in']
        args_dict['dec_in'] = self._resolved_dimensions['dec_in']
        args_dict['c_out'] = self._resolved_dimensions['c_out_model'] # Model's output layer size
        args_dict['quantile_levels'] = self._resolved_dimensions['quantile_levels'] # Pass quantile info
        # Add c_out_evaluation to args so models/mixins can access the base target count
        args_dict['c_out_evaluation'] = self._resolved_dimensions['c_out_evaluation']

        # Ensure target is a comma-separated string if it was a list from analysis
        if isinstance(args_dict.get('target'), list):
             args_dict['target'] = ','.join(args_dict['target'])

        # Remove keys that are specifically for the manager or not model args
        # (e.g., 'scaler' might be added to args temporarily elsewhere)
        args_dict.pop('scaler', None)

        # Return as Namespace
        return Namespace(**args_dict)

    def get_evaluation_info(self) -> Dict[str, Any]:
        """
        Returns information needed for loss calculation and evaluation metrics.
        """
        if not self._resolved_dimensions:
            raise RuntimeError("Dimensions must be resolved before getting evaluation info.")

        return {
            'c_out_evaluation': self._resolved_dimensions['c_out_evaluation'], # Number of base target features
            'num_quantiles': self._resolved_dimensions['num_quantiles'],
            'quantile_levels': self._resolved_dimensions['quantile_levels'],
            'loss_name': self._resolved_dimensions['loss_name'],
            'features_mode': self._resolved_dimensions['features_mode'],
            'task_name': self._resolved_dimensions['task_name'],
            'target_columns': self._resolved_dimensions['target_columns'],
            'n_targets_data': self._resolved_dimensions['n_targets_data'], # Number of target columns in raw data
            'n_total_features_data': self._resolved_dimensions['n_total_features_data'], # Total features in raw data
        }

    def get_data_analysis_results(self) -> Optional[Dict]:
        """Returns the raw results from dataset analysis."""
        return self._analysis_results

    def get_resolved_dimensions(self) -> Dict[str, Any]:
        """Returns the full dictionary of resolved dimensions."""
        return self._resolved_dimensions

    # Optional: Method to update a config file with resolved dimensions
    def update_config_file(self, original_config_path: str, output_config_path: str):
        """
        Loads an existing config, updates dimension fields, and saves to a new file.
        """
        if not self._resolved_dimensions:
            raise RuntimeError("Dimensions must be resolved before updating config file.")

        try:
            with open(original_config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Original config file not found: {original_config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading original config file: {e}")
            raise

        # Update dimension fields
        config['enc_in'] = self._resolved_dimensions['enc_in']
        config['dec_in'] = self._resolved_dimensions['dec_in']
        config['c_out'] = self._resolved_dimensions['c_out_model'] # Update c_out to model output size
        # Add c_out_evaluation to config for clarity/downstream use
        config['c_out_evaluation'] = self._resolved_dimensions['c_out_evaluation']

        # Optionally update other related fields
        config['features'] = self._resolved_dimensions['features_mode']
        config['task_name'] = self._resolved_dimensions['task_name']
        config['loss'] = self._resolved_dimensions['loss_name']
        if self._resolved_dimensions['quantile_levels'] is not None:
             config['quantile_levels'] = self._resolved_dimensions['quantile_levels']
             # Ensure loss is set to pinball/quantile if quantiles are used
             if config.get('loss', '').lower() not in ['pinball', 'quantile']:
                  config['loss'] = 'pinball' # Or 'quantile' depending on your convention
                  logger.warning(f"Config loss was not pinball/quantile but quantile_levels were set. Updated loss to '{config['loss']}'.")

        # Save the updated config
        os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated config saved to {output_config_path}")

    # Optional: Method to validate a config against resolved dimensions
    def validate_config(self, config: Dict[str, Any]) -> Dict:
        """
        Validates a config dictionary against the currently resolved dimensions.
        """
        if not self._resolved_dimensions:
            raise RuntimeError("Dimensions must be resolved before validating config.")

        validation = {
            'valid': True,
            'issues': []
        }

        checks = [
            ('enc_in', self._resolved_dimensions['enc_in']),
            ('dec_in', self._resolved_dimensions['dec_in']),
            ('c_out', self._resolved_dimensions['c_out_model']), # Check against model output size
            ('features', self._resolved_dimensions['features_mode']),
            ('task_name', self._resolved_dimensions['task_name']),
            ('loss', self._resolved_dimensions['loss_name']),
            # Add checks for quantile_levels if needed
        ]

        for param, expected_value in checks:
            config_value = config.get(param)
            # Handle case-insensitivity for strings like loss name
            if isinstance(config_value, str) and isinstance(expected_value, str):
                 if config_value.lower() != expected_value.lower():
                      validation['valid'] = False
                      validation['issues'].append({
                          'parameter': param,
                          'config_value': config_value,
                          'expected_value': expected_value,
                          'description': f"Config has {param}='{config_value}', but resolved dimensions expect '{expected_value}'"
                      })
            elif config_value != expected_value:
                validation['valid'] = False
                validation['issues'].append({
                    'parameter': param,
                    'config_value': config_value,
                    'expected_value': expected_value,
                    'description': f"Config has {param}={config_value}, but resolved dimensions expect {expected_value}"
                })

        # Check quantile levels specifically
        config_q_levels = config.get('quantile_levels')
        resolved_q_levels = self._resolved_dimensions['quantile_levels']

        if (config_q_levels is None and resolved_q_levels is not None) or \
           (config_q_levels is not None and resolved_q_levels is None) or \
           (config_q_levels is not None and resolved_q_levels is not None and sorted(config_q_levels) != sorted(resolved_q_levels)):
             validation['valid'] = False
             validation['issues'].append({
                 'parameter': 'quantile_levels',
                 'config_value': config_q_levels,
                 'expected_value': resolved_q_levels,
                 'description': f"Config has quantile_levels={config_q_levels}, but resolved dimensions expect {resolved_q_levels}"
             })

        # Check c_out_evaluation if present in config
        if 'c_out_evaluation' in config:
             if config['c_out_evaluation'] != self._resolved_dimensions['c_out_evaluation']:
                  validation['valid'] = False
                  validation['issues'].append({
                      'parameter': 'c_out_evaluation',
                      'config_value': config['c_out_evaluation'],
                      'expected_value': self._resolved_dimensions['c_out_evaluation'],
                      'description': f"Config has c_out_evaluation={config['c_out_evaluation']}, but resolved dimensions expect {self._resolved_dimensions['c_out_evaluation']}"
                  })

        return validation

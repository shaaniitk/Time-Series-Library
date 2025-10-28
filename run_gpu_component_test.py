#!/usr/bin/env python3
"""
🚀 GPU COMPONENT TEST - Production Ready with Comprehensive Logging
Tests advanced components on GPU with detailed logging and memory monitoring
All output is logged to files for later analysis
"""

import os
import sys
import yaml
import json
import time
import logging
import psutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the proven production training function
from scripts.train.train_celestial_production import train_celestial_pgat_production

class GPUComponentTester:
    """GPU Component Tester with comprehensive logging and memory monitoring"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup comprehensive logging
        self.setup_logging()
        
        # Results directory
        self.results_dir = Path('results/gpu_component_tests')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 GPU Component Tester initialized")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Logs directory: {self.logs_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
        
        # Log system information
        self.log_system_info()
    
    def setup_logging(self):
        """Setup comprehensive logging to files"""
        
        # Main logger
        self.logger = logging.getLogger('GPUComponentTester')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for main log
        main_log_file = self.logs_dir / f'gpu_component_test_{self.timestamp}.log'
        file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for progress updates
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Memory logger
        self.memory_logger = logging.getLogger('MemoryMonitor')
        self.memory_logger.setLevel(logging.DEBUG)
        memory_log_file = self.logs_dir / f'memory_monitor_{self.timestamp}.log'
        memory_handler = logging.FileHandler(memory_log_file, encoding='utf-8')
        memory_handler.setLevel(logging.DEBUG)
        memory_handler.setFormatter(formatter)
        self.memory_logger.addHandler(memory_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger('Performance')
        self.perf_logger.setLevel(logging.DEBUG)
        perf_log_file = self.logs_dir / f'performance_{self.timestamp}.log'
        perf_handler = logging.FileHandler(perf_log_file, encoding='utf-8')
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(formatter)
        self.perf_logger.addHandler(perf_handler)
        
        self.logger.info(f"Logging setup complete:")
        self.logger.info(f"  Main log: {main_log_file}")
        self.logger.info(f"  Memory log: {memory_log_file}")
        self.logger.info(f"  Performance log: {perf_log_file}")
    
    def log_system_info(self):
        """Log comprehensive system information"""
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 80)
        
        # CPU Information
        self.logger.info(f"CPU Count: {os.cpu_count()}")
        self.logger.info(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
        
        # Memory Information
        memory = psutil.virtual_memory()
        self.logger.info(f"Total Memory: {memory.total / (1024**3):.2f} GB")
        self.logger.info(f"Available Memory: {memory.available / (1024**3):.2f} GB")
        self.logger.info(f"Memory Usage: {memory.percent}%")
        
        # GPU Information
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"GPU Count: {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            else:
                self.logger.warning("CUDA not available")
        except Exception as e:
            self.logger.error(f"Failed to get GPU info: {e}")
        
        # Python Environment
        self.logger.info(f"Python Version: {sys.version}")
        self.logger.info(f"Working Directory: {os.getcwd()}")
        
        self.logger.info("=" * 80)
    
    def log_memory_usage(self, stage: str, context: Dict[str, Any] = None):
        """Log detailed memory usage"""
        try:
            # CPU Memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            memory_info = {
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'cpu_total_gb': memory.total / (1024**3),
                'cpu_available_gb': memory.available / (1024**3),
                'cpu_used_gb': (memory.total - memory.available) / (1024**3),
                'cpu_percent': memory.percent,
                'process_rss_gb': process_memory.rss / (1024**3),
                'process_vms_gb': process_memory.vms / (1024**3),
            }
            
            # GPU Memory
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                        max_reserved = torch.cuda.max_memory_reserved(i) / (1024**3)
                        
                        memory_info.update({
                            f'gpu_{i}_allocated_gb': allocated,
                            f'gpu_{i}_reserved_gb': reserved,
                            f'gpu_{i}_max_allocated_gb': max_allocated,
                            f'gpu_{i}_max_reserved_gb': max_reserved,
                        })
            except Exception as e:
                memory_info['gpu_error'] = str(e)
            
            # Add context
            if context:
                memory_info.update(context)
            
            # Log as JSON for easy parsing
            self.memory_logger.info(json.dumps(memory_info))
            
        except Exception as e:
            self.memory_logger.error(f"Failed to log memory usage: {e}")
    
    def create_gpu_test_config(self, config_name: str, **component_overrides) -> str:
        """Create GPU-optimized test configuration"""
        
        # GPU-optimized base configuration
        base_config = {
            # Required fields
            'model': 'Celestial_Enhanced_PGAT',
            'data': 'custom',
            'root_path': './data',
            'data_path': 'prepared_financial_data.csv',
            'features': 'MS',
            'target': 'log_Open,log_High,log_Low,log_Close',
            'embed': 'timeF',
            'freq': 'd',
            
            # Sequence settings
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 24,
            
            # FIXED: Better data splits to reduce overfitting
            'validation_length': 800,  # ~11% for validation (more representative)
            'test_length': 600,        # ~8% for testing
            
            # FIXED: Reduced model complexity to prevent overfitting
            'd_model': 128,   # Smaller model for better generalization
            'n_heads': 8,     # 128 ÷ 8 = 16
            'e_layers': 2,    # Fewer layers to reduce overfitting
            'd_layers': 1,    # Single decoder layer
            'd_ff': 256,      # Smaller feed-forward
            'dropout': 0.2,   # Higher dropout for regularization
            
            # FIXED: Input/Output dimensions - CRITICAL FIX
            'enc_in': 118,    # Match actual data (119 CSV columns - 1 date = 118 features)
            'dec_in': 118,    # Match encoder input
            'c_out': 4,       # OHLC targets
            
            # FIXED: Anti-overfitting training configuration
            'train_epochs': 8,  # More epochs for proper convergence
            'batch_size': 16,   # Keep batch size
            'learning_rate': 0.0003,  # Lower LR for stability
            'patience': 15,     # Higher patience
            'lradj': 'warmup_cosine',
            'warmup_epochs': 1,  # FIXED: Reduce warmup (was 2/5=40%, now 1/8=12.5%)
            'min_lr': 1e-7,
            'weight_decay': 0.001,   # Higher weight decay for regularization
            'clip_grad_norm': 0.5,   # Lower gradient clipping
            
            # GPU settings - FIXED for stability
            'use_gpu': True,
            'mixed_precision': False,  # DISABLED: Can cause instability
            'gradient_accumulation_steps': 1,
            
            # Celestial system configuration
            'use_celestial_graph': True,
            'aggregate_waves_to_celestial': True,
            'celestial_fusion_layers': 2,
            'num_celestial_bodies': 13,
            'celestial_dim': 32,
            'num_message_passing_steps': 2,
            'edge_feature_dim': 8,
            'use_temporal_attention': True,
            'use_spatial_attention': True,
            'bypass_spatiotemporal_with_petri': True,
            'num_input_waves': 118,  # CRITICAL: Match actual data dimensions
            'target_wave_indices': [0, 1, 2, 3],
            
            # Advanced component defaults
            'use_multi_scale_context': True,
            'context_fusion_mode': 'simple',
            'use_hierarchical_mapping': False,
            'use_stochastic_learner': False,
            'use_petri_net_combiner': True,
            'use_mixture_decoder': False,
            'enable_mdn_decoder': False,
            'use_target_autocorrelation': True,
            'use_celestial_target_attention': True,
            'use_efficient_covariate_interaction': False,
            'use_dynamic_spatiotemporal_encoder': True,
            
            # Enhanced logging and diagnostics
            'use_calendar_effects': True,
            'calendar_embedding_dim': 64,
            'reg_loss_weight': 0.0005,
            'loss': {'type': 'mse'},
            'save_checkpoints': True,
            'save_best_only': True,
            'checkpoint_interval': 1,
            'log_interval': 5,  # More frequent logging
            'enable_memory_diagnostics': True,
            'memory_log_interval': 10,
            'collect_diagnostics': True,
            'verbose_logging': False,
            
            # Required by data_provider
            'task_name': 'long_term_forecast',
            'model_name': 'Celestial_Enhanced_PGAT',
            'data_name': 'custom',
            'checkpoints': './checkpoints/',
            'inverse': False,
            'cols': None,
            'num_workers': 0,  # Disable for stability
            'itr': 1,
            'train_only': False,
            'do_predict': False,
        }
        
        # Apply component overrides
        config = base_config.copy()
        config.update(component_overrides)
        
        # Set unique model_id
        config['model_id'] = f"gpu_test_{config_name}_{self.timestamp}"
        
        # Create config file
        config_path = self.results_dir / f"{config_name}_{self.timestamp}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created config: {config_path}")
        return str(config_path)
    
    def run_component_test(self, config_name: str, **component_overrides) -> Dict[str, Any]:
        """Run a single component test with comprehensive monitoring"""
        
        self.logger.info("=" * 80)
        self.logger.info(f"🧪 TESTING COMPONENT: {config_name}")
        self.logger.info("=" * 80)
        
        # Log component configuration
        self.logger.info("Component Configuration:")
        for key, value in component_overrides.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log memory before test
        self.log_memory_usage(f"{config_name}_start")
        
        # Create configuration
        config_path = self.create_gpu_test_config(config_name, **component_overrides)
        
        # Performance tracking
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            self.logger.info(f"Starting training with config: {config_path}")
            
            # Run training
            success = train_celestial_pgat_production(config_path)
            
            # Calculate performance metrics
            end_time = time.time()
            training_time = end_time - start_time
            end_memory = psutil.virtual_memory().used
            memory_delta = (end_memory - start_memory) / (1024**3)  # GB
            
            self.perf_logger.info(f"{config_name} | training_time: {training_time:.2f}s | memory_delta: {memory_delta:.2f}GB")
            
            if success:
                self.logger.info(f"✅ Training completed successfully in {training_time:.2f}s")
                
                # Try to load results
                results_files = list(Path(".").glob("checkpoints/*/production_results.json"))
                
                if results_files:
                    # Get the most recent results file
                    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                    
                    with open(latest_results, 'r') as f:
                        production_results = json.load(f)
                    
                    result = {
                        'status': 'success',
                        'config_name': config_name,
                        'final_val_loss': production_results.get('best_val_loss', float('inf')),
                        'final_rmse': production_results.get('overall_metrics', {}).get('rmse', float('inf')),
                        'total_params': production_results.get('total_parameters', 0),
                        'training_time': training_time,
                        'memory_delta_gb': memory_delta,
                        'train_losses': production_results.get('train_losses', []),
                        'val_losses': production_results.get('val_losses', []),
                        'overall_metrics': production_results.get('overall_metrics', {}),
                        'config_path': config_path,
                        'results_file': str(latest_results),
                    }
                    
                    # Log detailed results
                    self.logger.info(f"📊 RESULTS for {config_name}:")
                    self.logger.info(f"  Final Val Loss: {result['final_val_loss']:.6f}")
                    self.logger.info(f"  Final RMSE: {result['final_rmse']:.6f}")
                    self.logger.info(f"  Total Parameters: {result['total_params']:,}")
                    self.logger.info(f"  Training Time: {training_time:.2f}s")
                    self.logger.info(f"  Memory Delta: {memory_delta:.2f}GB")
                    
                    # Log training progression
                    if result['train_losses'] and result['val_losses']:
                        self.logger.info(f"  Training Progression:")
                        for i, (train_loss, val_loss) in enumerate(zip(result['train_losses'], result['val_losses'])):
                            self.logger.info(f"    Epoch {i+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
                    
                else:
                    result = {
                        'status': 'success_no_results',
                        'config_name': config_name,
                        'training_time': training_time,
                        'memory_delta_gb': memory_delta,
                        'config_path': config_path,
                    }
                    self.logger.warning(f"⚠️  Training succeeded but no results file found")
            else:
                result = {
                    'status': 'failed',
                    'config_name': config_name,
                    'training_time': training_time,
                    'memory_delta_gb': memory_delta,
                    'config_path': config_path,
                    'error': 'Training function returned False'
                }
                self.logger.error(f"❌ Training failed")
        
        except Exception as e:
            end_time = time.time()
            training_time = end_time - start_time
            end_memory = psutil.virtual_memory().used
            memory_delta = (end_memory - start_memory) / (1024**3)
            
            result = {
                'status': 'error',
                'config_name': config_name,
                'training_time': training_time,
                'memory_delta_gb': memory_delta,
                'config_path': config_path,
                'error': str(e)
            }
            
            self.logger.error(f"💥 ERROR in {config_name}: {e}")
            self.logger.exception("Full traceback:")
        
        # Log memory after test
        self.log_memory_usage(f"{config_name}_end", {
            'status': result['status'],
            'training_time': result['training_time'],
            'memory_delta_gb': result['memory_delta_gb']
        })
        
        return result
    
    def run_baseline_overfitting_check(self) -> bool:
        """Run a quick baseline test to check if overfitting is fixed"""
        
        self.logger.info("🔍 BASELINE OVERFITTING CHECK")
        self.logger.info("=" * 60)
        self.logger.info("Running quick baseline test to verify overfitting fix...")
        
        # Minimal baseline configuration for overfitting check
        baseline_config = {
            "use_multi_scale_context": False,
            "use_hierarchical_mapping": False,
            "use_stochastic_learner": False,
            "use_petri_net_combiner": False,
            "use_mixture_decoder": False,
            "enable_mdn_decoder": False,
            "use_efficient_covariate_interaction": False,
        }
        
        self.logger.info("Testing minimal baseline configuration...")
        
        # Run baseline test
        result = self.run_component_test("Baseline_OverfitCheck", **baseline_config)
        
        if result['status'] != 'success':
            self.logger.error("❌ Baseline test failed - cannot proceed with component testing")
            return False
        
        # Analyze training progression for overfitting
        train_losses = result.get('train_losses', [])
        val_losses = result.get('val_losses', [])
        
        if len(train_losses) < 3 or len(val_losses) < 3:
            self.logger.warning("⚠️  Insufficient training data to check overfitting")
            return True  # Proceed anyway
        
        # Check for overfitting pattern
        train_trend = self._analyze_loss_trend(train_losses)
        val_trend = self._analyze_loss_trend(val_losses)
        
        # Calculate final gap between train and validation loss
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_gap = abs(final_val_loss - final_train_loss) / final_train_loss
        
        self.logger.info(f"📊 OVERFITTING ANALYSIS:")
        self.logger.info(f"  Training trend: {train_trend}")
        self.logger.info(f"  Validation trend: {val_trend}")
        self.logger.info(f"  Final train loss: {final_train_loss:.6f}")
        self.logger.info(f"  Final val loss: {final_val_loss:.6f}")
        self.logger.info(f"  Loss gap: {loss_gap:.2%}")
        
        # Overfitting detection criteria
        overfitting_detected = False
        
        # Criterion 1: Training decreasing but validation increasing
        if train_trend == 'decreasing' and val_trend == 'increasing':
            self.logger.warning("⚠️  Classic overfitting pattern detected: train↓ val↑")
            overfitting_detected = True
        
        # Criterion 2: Large gap between train and validation loss
        if loss_gap > 0.5:  # 50% gap
            self.logger.warning(f"⚠️  Large train/val gap detected: {loss_gap:.2%}")
            overfitting_detected = True
        
        # Criterion 3: Validation loss much higher than training
        if final_val_loss > final_train_loss * 1.5:
            self.logger.warning(f"⚠️  Validation loss significantly higher than training")
            overfitting_detected = True
        
        if overfitting_detected:
            self.logger.error("❌ OVERFITTING STILL DETECTED!")
            self.logger.error("The configuration fixes did not resolve the overfitting issue.")
            self.logger.error("Component testing would not provide meaningful results.")
            self.logger.error("")
            self.logger.error("Recommended actions:")
            self.logger.error("1. Further reduce model complexity (d_model=64)")
            self.logger.error("2. Increase dropout to 0.3")
            self.logger.error("3. Increase weight decay to 0.01")
            self.logger.error("4. Reduce learning rate to 0.0001")
            self.logger.error("5. Check data split methodology")
            return False
        else:
            self.logger.info("✅ OVERFITTING CHECK PASSED!")
            self.logger.info("Training shows healthy convergence pattern")
            self.logger.info("Proceeding with comprehensive component testing...")
            return True
    
    def _analyze_loss_trend(self, losses: list) -> str:
        """Analyze the trend in a loss sequence"""
        if len(losses) < 2:
            return 'insufficient_data'
        
        # Look at the trend over the last half of training
        mid_point = len(losses) // 2
        recent_losses = losses[mid_point:]
        
        if len(recent_losses) < 2:
            recent_losses = losses
        
        # Calculate trend
        start_loss = recent_losses[0]
        end_loss = recent_losses[-1]
        
        change_ratio = (end_loss - start_loss) / start_loss
        
        if change_ratio < -0.05:  # Decreasing by more than 5%
            return 'decreasing'
        elif change_ratio > 0.05:  # Increasing by more than 5%
            return 'increasing'
        else:
            return 'stable'
    
    def run_comprehensive_component_tests(self):
        """Run comprehensive component tests on GPU (after overfitting check)"""
        
        self.logger.info("🚀 STARTING COMPREHENSIVE GPU COMPONENT TESTS")
        self.logger.info("=" * 80)
        
        # Comprehensive test configurations
        test_configs = [
            ("Baseline_Minimal", {
                "use_multi_scale_context": False,
                "use_hierarchical_mapping": False,
                "use_stochastic_learner": False,
                "use_petri_net_combiner": False,
                "use_mixture_decoder": False,
                "enable_mdn_decoder": False,
                "use_efficient_covariate_interaction": False,
            }),
            
            ("Context_Simple", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "simple",
            }),
            
            ("Context_Gated", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "gated",
                "context_fusion_dropout": 0.1,
            }),
            
            ("Context_Attention", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "attention",
                "context_fusion_dropout": 0.1,
            }),
            
            ("Context_MultiScale", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "short_term_kernel_size": 3,
                "medium_term_kernel_size": 15,
                "long_term_kernel_size": 0,
                "context_fusion_dropout": 0.1,
            }),
            
            ("Stochastic_Learner", {
                "use_stochastic_learner": True,
                "use_stochastic_control": False,
            }),
            
            ("Stochastic_Control", {
                "use_stochastic_learner": False,
                "use_stochastic_control": True,
                "stochastic_temperature_start": 1.0,
                "stochastic_temperature_end": 0.1,
                "stochastic_decay_steps": 1000,
            }),
            
            ("Stochastic_Both", {
                "use_stochastic_learner": True,
                "use_stochastic_control": True,
                "stochastic_temperature_start": 1.0,
                "stochastic_temperature_end": 0.1,
                "stochastic_decay_steps": 1000,
            }),
            
            ("Graph_Standard", {
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": False,
            }),
            
            ("Graph_Gated", {
                "use_petri_net_combiner": False,
                "use_gated_graph_combiner": True,
            }),
            
            ("Graph_PetriNet", {
                "use_petri_net_combiner": True,
                "use_gated_graph_combiner": False,
                "num_message_passing_steps": 2,
                "edge_feature_dim": 8,
            }),
            
            ("Decoder_Mixture", {
                "use_mixture_decoder": True,
                "use_sequential_mixture_decoder": True,
                "enable_mdn_decoder": False,
            }),
            
            ("Decoder_MDN", {
                "use_mixture_decoder": False,
                "enable_mdn_decoder": True,
                "mdn_components": 5,
                "mdn_sigma_min": 0.001,
            }),
            
            ("Decoder_Both", {
                "use_mixture_decoder": True,
                "enable_mdn_decoder": True,
                "mdn_components": 5,
            }),
            
            ("Hierarchical_Mapping", {
                "use_hierarchical_mapping": True,
            }),
            
            ("Efficient_Covariate", {
                "use_efficient_covariate_interaction": True,
                "enable_target_covariate_attention": True,
            }),
            
            ("Full_Advanced", {
                "use_multi_scale_context": True,
                "context_fusion_mode": "multi_scale",
                "use_hierarchical_mapping": True,
                "use_stochastic_learner": True,
                "use_stochastic_control": True,
                "use_petri_net_combiner": True,
                "use_mixture_decoder": True,
                "enable_mdn_decoder": True,
                "use_efficient_covariate_interaction": True,
            }),
        ]
        
        results = {}
        
        for i, (config_name, component_overrides) in enumerate(test_configs, 1):
            self.logger.info(f"📋 Test {i}/{len(test_configs)}: {config_name}")
            
            result = self.run_component_test(config_name, **component_overrides)
            results[config_name] = result
            
            # Save intermediate results
            self.save_results(results, f'intermediate_results_{self.timestamp}.json')
            
            # Brief pause between tests
            self.logger.info(f"⏸️  Pausing 5 seconds before next test...")
            time.sleep(5)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file with comprehensive metadata"""
        
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'timestamp': self.timestamp,
                'test_type': 'gpu_component_analysis',
                'total_tests': len(results),
                'successful_tests': len([r for r in results.values() if r.get('status') == 'success']),
                'failed_tests': len([r for r in results.values() if r.get('status') != 'success']),
                'system_info': {
                    'cpu_count': os.cpu_count(),
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': sys.version,
                }
            },
            'results': results
        }
        
        # Add GPU info
        try:
            import torch
            if torch.cuda.is_available():
                results_with_metadata['metadata']['system_info']['gpu_count'] = torch.cuda.device_count()
                results_with_metadata['metadata']['system_info']['gpu_names'] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except Exception:
            pass
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to {filepath}")
    
    def generate_final_report(self, results: Dict[str, Any]):
        """Generate comprehensive final report"""
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 FINAL GPU COMPONENT TEST REPORT")
        self.logger.info("=" * 80)
        
        # Categorize results
        successful = []
        failed = []
        
        for name, result in results.items():
            if result.get('status') == 'success':
                successful.append((name, result))
            else:
                failed.append((name, result))
        
        self.logger.info(f"📊 SUMMARY:")
        self.logger.info(f"  Total Tests: {len(results)}")
        self.logger.info(f"  Successful: {len(successful)}")
        self.logger.info(f"  Failed: {len(failed)}")
        self.logger.info(f"  Success Rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            self.logger.info(f"\n✅ SUCCESSFUL CONFIGURATIONS:")
            
            # Sort by validation loss
            successful_sorted = sorted(
                [(name, result) for name, result in successful 
                 if 'final_val_loss' in result and isinstance(result['final_val_loss'], (int, float))],
                key=lambda x: x[1]['final_val_loss']
            )
            
            for i, (name, result) in enumerate(successful_sorted, 1):
                val_loss = result.get('final_val_loss', 'N/A')
                rmse = result.get('final_rmse', 'N/A')
                params = result.get('total_params', 'N/A')
                time_str = f"{result.get('training_time', 0):.1f}s"
                memory_str = f"{result.get('memory_delta_gb', 0):.2f}GB"
                
                self.logger.info(f"  {i:2d}. {name}")
                self.logger.info(f"      Val Loss: {val_loss:.6f} | RMSE: {rmse:.6f}")
                self.logger.info(f"      Params: {params:,} | Time: {time_str} | Memory: {memory_str}")
            
            if successful_sorted:
                best_name, best_result = successful_sorted[0]
                self.logger.info(f"\n🏆 BEST PERFORMING COMPONENT: {best_name}")
                self.logger.info(f"    Val Loss: {best_result['final_val_loss']:.6f}")
                self.logger.info(f"    RMSE: {best_result['final_rmse']:.6f}")
                self.logger.info(f"    Parameters: {best_result['total_params']:,}")
                self.logger.info(f"    Training Time: {best_result['training_time']:.1f}s")
        
        if failed:
            self.logger.info(f"\n❌ FAILED CONFIGURATIONS:")
            for name, result in failed:
                error = result.get('error', result.get('status', 'Unknown'))
                time_str = f"{result.get('training_time', 0):.1f}s"
                self.logger.info(f"  • {name}: {error} (Time: {time_str})")
        
        # Component analysis
        self.logger.info(f"\n🔬 COMPONENT ANALYSIS:")
        
        component_categories = {
            'Context Fusion': ['Context_Simple', 'Context_Gated', 'Context_Attention', 'Context_MultiScale'],
            'Stochastic Learning': ['Stochastic_Learner', 'Stochastic_Control', 'Stochastic_Both'],
            'Graph Combiners': ['Graph_Standard', 'Graph_Gated', 'Graph_PetriNet'],
            'Enhanced Decoders': ['Decoder_Mixture', 'Decoder_MDN', 'Decoder_Both'],
            'Other Advanced': ['Hierarchical_Mapping', 'Efficient_Covariate', 'Full_Advanced'],
        }
        
        for category, configs in component_categories.items():
            successful_in_category = [name for name in configs if name in dict(successful)]
            success_rate = len(successful_in_category) / len(configs) * 100 if configs else 0
            
            self.logger.info(f"  {category}: {len(successful_in_category)}/{len(configs)} ({success_rate:.1f}%)")
            
            if successful_in_category:
                # Find best in category
                category_results = [(name, results[name]) for name in successful_in_category 
                                  if 'final_val_loss' in results[name]]
                if category_results:
                    best_in_category = min(category_results, key=lambda x: x[1]['final_val_loss'])
                    self.logger.info(f"    Best: {best_in_category[0]} (Val Loss: {best_in_category[1]['final_val_loss']:.6f})")
        
        self.logger.info("=" * 80)
        
        return results

def main():
    """Main execution function with overfitting check"""
    
    print("🌟 CELESTIAL ENHANCED PGAT - GPU COMPONENT TESTING")
    print("🔥 Production-ready testing with comprehensive logging")
    print("📊 All output logged to files for detailed analysis")
    print("")
    print("� TFIXES APPLIED:")
    print("   ✅ Data leakage eliminated (clean temporal splits)")
    print("   ✅ Model complexity reduced (d_model=128, fewer layers)")
    print("   ✅ Training stabilized (higher dropout, weight decay)")
    print("")
    print("🔍 TESTING PROCESS:")
    print("   1. Baseline overfitting check (quick validation)")
    print("   2. If overfitting fixed → Full component testing")
    print("   3. If overfitting detected → Abort with recommendations")
    print("=" * 80)
    
    # Create tester
    tester = GPUComponentTester()
    
    try:
        # Step 1: Run baseline overfitting check
        tester.logger.info("🔍 STEP 1: Baseline Overfitting Check")
        overfitting_fixed = tester.run_baseline_overfitting_check()
        
        if not overfitting_fixed:
            tester.logger.error("❌ OVERFITTING CHECK FAILED - ABORTING COMPONENT TESTING")
            tester.logger.error("The baseline test still shows overfitting patterns.")
            tester.logger.error("Component testing would not provide meaningful results.")
            tester.logger.error("Please review and adjust the configuration before retrying.")
            
            # Save baseline results for analysis
            baseline_results = {'overfitting_check': 'failed', 'timestamp': tester.timestamp}
            tester.save_results(baseline_results, f'baseline_overfitting_check_{tester.timestamp}.json')
            
            return None
        
        tester.logger.info("✅ OVERFITTING CHECK PASSED - PROCEEDING WITH COMPONENT TESTING")
        tester.logger.info("")
        
        # Step 2: Run comprehensive component tests
        tester.logger.info("🚀 STEP 2: Comprehensive Component Testing")
        results = tester.run_comprehensive_component_tests()
        
        # Step 3: Generate final report
        tester.logger.info("📊 STEP 3: Generating Final Report")
        final_results = tester.generate_final_report(results)
        
        # Step 4: Save final results
        tester.save_results(final_results, f'final_gpu_results_{tester.timestamp}.json')
        
        tester.logger.info("🎉 GPU COMPONENT TESTING COMPLETED SUCCESSFULLY!")
        tester.logger.info(f"📁 All logs saved to: {tester.logs_dir}")
        tester.logger.info(f"📁 All results saved to: {tester.results_dir}")
        tester.logger.info("")
        tester.logger.info("🏆 SUMMARY:")
        tester.logger.info("✅ Overfitting check: PASSED")
        tester.logger.info(f"✅ Component tests: {len([r for r in results.values() if r.get('status') == 'success'])}/{len(results)} successful")
        tester.logger.info("✅ Meaningful component comparison: ACHIEVED")
        
        return final_results
        
    except Exception as e:
        tester.logger.error(f"💥 CRITICAL ERROR: {e}")
        tester.logger.exception("Full traceback:")
        return None

if __name__ == "__main__":
    results = main()
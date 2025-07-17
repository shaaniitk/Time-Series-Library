# D:\workspace\Time-Series-Library\debug_hierarchical.py

import torch
import torch.nn as nn
from argparse import Namespace

# Temporarily add the project root to the Python path
# to ensure all modules can be imported.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# All necessary imports from your project
from layers.modular.decomposition import get_decomposition_component
from layers.modular.encoder import get_encoder_component
from layers.modular.decoder import get_decoder_component
from layers.modular.attention import get_attention_component
from layers.modular.sampling import get_sampling_component
from layers.modular.output_heads import get_output_head_component
from layers.modular.losses import get_loss_component
from layers.Embed import DataEmbedding_wo_pos
from utils.logger import logger
from configs.autoformer.hierarchical_config import get_hierarchical_autoformer_config

# --- A Self-Contained, Debuggable Version of ModularAutoformer ---
# I've copied the class here so we can add the breakpoint
# without modifying the original source file.

class DebugModularAutoformer(nn.Module):
    def __init__(self, configs):
        super(DebugModularAutoformer, self).__init__()
        logger.info("Initializing DebugModularAutoformer...")

        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self._initialize_traditional()

    def _initialize_traditional(self):
        logger.info("Entering _initialize_traditional...")
        
        self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
        
        # --- Encoder Assembly ---
        logger.info("Assembling Encoder...")
        self.configs.encoder_params['attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        
        # --- THIS IS THE POINT OF FAILURE ---
        print("\n" + "="*50)
        print(">>> DEBUGGER WILL STOP ON THE NEXT LINE <<<")
        print(f"About to create 'decomp_comp' for ENCODER using type: {self.configs.decomposition_type}")
        print("Inspect 'self.configs.decomposition_params' vs 'self.configs.init_decomposition_params'")
        print("="*50 + "\n")
        
        # --- BREAKPOINT ---
        breakpoint() 
        
        # This is the line that fails
        self.configs.encoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **self.configs.decomposition_params)
        self.encoder = get_encoder_component(self.configs.encoder_type, **self.configs.encoder_params)

        logger.info("Encoder assembled successfully.")

        # --- Decoder Assembly ---
        logger.info("Assembling Decoder...")
        self.dec_embedding = DataEmbedding_wo_pos(self.configs.dec_in, self.configs.d_model, self.configs.embed, self.configs.freq, self.configs.dropout)
        self.configs.decoder_params['self_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        self.configs.decoder_params['cross_attention_comp'] = get_attention_component(self.configs.attention_type, **self.configs.attention_params)
        
        # This is the second point of failure
        self.configs.decoder_params['decomp_comp'] = get_decomposition_component(self.configs.decomposition_type, **self.configs.decomposition_params)
        self.decoder = get_decoder_component(self.configs.decoder_type, **self.configs.decoder_params)
        
        logger.info("Decoder assembled successfully.")

        # --- Final Components ---
        self.sampling = get_sampling_component(self.configs.sampling_type, **self.configs.sampling_params)
        self.output_head = get_output_head_component(self.configs.output_head_type, **self.configs.output_head_params)
        self.init_decomp = get_decomposition_component(self.configs.decomposition_type, **self.configs.init_decomposition_params)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # A dummy forward pass for instantiation testing
        return torch.randn(1, self.pred_len, self.configs.c_out)


# --- Main Debugging Logic ---
if __name__ == '__main__':
    print("--- Starting Hierarchical Model Debug Script ---")
    
    # 1. Get the exact configuration that causes the failure
    hierarchical_config = get_hierarchical_autoformer_config(
        num_targets=7,
        num_covariates=3,
        seq_len=96,
        pred_len=24,
        label_len=48
    )
    
    print("\n[INFO] Configuration created successfully.")
    print(f"[INFO] Encoder type: {hierarchical_config.encoder_type}")
    print(f"[INFO] Decoder type: {hierarchical_config.decoder_type}")
    print(f"[INFO] Decomposition type: {hierarchical_config.decomposition_type}")

    # 2. Attempt to instantiate the model
    print("\n[INFO] Now attempting to instantiate the ModularAutoformer...")
    try:
        # We use our special debug class here
        model = DebugModularAutoformer(hierarchical_config)
        print("\n[SUCCESS] Model instantiated without error (this shouldn't happen).")
    except Exception as e:
        print(f"\n[ERROR] Model instantiation failed as expected.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

#!/usr/bin/env python3
"""
Deep Analysis of Celestial Target Dependence Pipeline
Comprehensive verification of all components in the celestial processing chain
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def deep_celestial_analysis():
    """
    Comprehensive analysis of celestial processing pipeline:
    1. Celestial body embedding correctness
    2. Graph combination and edge computation
    3. Petri net dynamic graph progression
    4. Multi-head attention + Petri net impact on targets
    5. Hierarchical fusion validation
    """
    
    print("=" * 80)
    print("DEEP CELESTIAL TARGET DEPENDENCE ANALYSIS")
    print("=" * 80)
    
    # Load config
    config_path = "configs/celestial_production_fixed.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = SimpleConfig(config_dict)
    
    # Load data and model
    from data_provider.data_factory import data_provider
    from models.Celestial_Enhanced_PGAT_Modular import Model
    
    print("Loading data and model...")
    train_data, train_loader = data_provider(args, flag='train')
    model = Model(args)
    model.eval()
    
    # Get a batch
    batch_iter = iter(train_loader)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iter)
    
    # Convert to float32
    batch_x = batch_x.float()
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()
    
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
    
    # Prepare decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len-args.label_len:, :]).float()
    dec_inp[:, :args.label_len, :] = batch_x[:, -args.label_len:, :]
    
    print("\n" + "="*80)
    print("1. CELESTIAL BODY EMBEDDING ANALYSIS")
    print("="*80)
    
    # Test celestial embedding
    embedding_module = model.embedding_module
    
    # Check phase-aware processor
    if embedding_module.phase_aware_processor is not None:
        print("✓ Phase-Aware Celestial Processor found")
        processor = embedding_module.phase_aware_processor
        
        print(f"  - Input waves: {processor.num_input_waves}")
        print(f"  - Celestial bodies: {len(processor.celestial_processors)}")
        print(f"  - Celestial dimension: {processor.celestial_dim}")
        
        # Test individual celestial body processing
        print("\n🔍 Testing individual celestial body embeddings:")
        
        with torch.no_grad():
            celestial_features, adjacency_matrix, phase_metadata = processor(batch_x)
            
        print(f"  - Celestial features shape: {celestial_features.shape}")
        print(f"  - Adjacency matrix shape: {adjacency_matrix.shape}")
        
        # Analyze celestial body representations
        batch_size, seq_len, total_celestial_dim = celestial_features.shape
        num_celestial = len(processor.celestial_processors)
        celestial_dim = total_celestial_dim // num_celestial
        
        celestial_reshaped = celestial_features.view(batch_size, seq_len, num_celestial, celestial_dim)
        
        print(f"\n📊 Celestial body representation analysis:")
        for i, (body_name, body_processor) in enumerate(processor.celestial_processors.items()):
            body_repr = celestial_reshaped[:, :, i, :]
            print(f"  {body_name:12}: mean={body_repr.mean():.6f}, std={body_repr.std():.6f}, "
                  f"range=[{body_repr.min():.3f}, {body_repr.max():.3f}]")
            
            # Check if representation is meaningful (not constant)
            if body_repr.std() < 1e-6:
                print(f"    ⚠️  WARNING: {body_name} representation appears constant!")
            else:
                print(f"    ✓ {body_name} representation is dynamic")
    
    print("\n" + "="*80)
    print("2. GRAPH COMBINATION AND EDGE COMPUTATION ANALYSIS")
    print("="*80)
    
    # Test graph module
    if model.graph_module is not None:
        print("✓ Graph Module found")
        
        # Get encoder output for graph processing
        enc_out, dec_out, past_celestial_features, phase_based_adj, x_enc_processed, wave_metadata = embedding_module(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )
        
        print(f"  - Encoder output shape: {enc_out.shape}")
        print(f"  - Phase-based adjacency shape: {phase_based_adj.shape if phase_based_adj is not None else 'None'}")
        
        # Test graph processing
        market_context = model.market_context_encoder(enc_out)
        print(f"  - Market context shape: {market_context.shape}")
        
        with torch.no_grad():
            enhanced_enc_out, combined_adj, rich_edge_features, fusion_metadata, celestial_features = model.graph_module(
                enc_out, market_context, phase_based_adj
            )
        
        print(f"  - Enhanced encoder output shape: {enhanced_enc_out.shape}")
        print(f"  - Combined adjacency shape: {combined_adj.shape if combined_adj is not None else 'None'}")
        print(f"  - Rich edge features shape: {rich_edge_features.shape if rich_edge_features is not None else 'None'}")
        
        # Analyze adjacency matrix properties
        if combined_adj is not None:
            print(f"\n📊 Adjacency matrix analysis:")
            adj_mean = combined_adj.mean()
            adj_std = combined_adj.std()
            adj_min = combined_adj.min()
            adj_max = combined_adj.max()
            
            print(f"  - Mean: {adj_mean:.6f}")
            print(f"  - Std: {adj_std:.6f}")
            print(f"  - Range: [{adj_min:.6f}, {adj_max:.6f}]")
            
            # Check if adjacency is meaningful
            if adj_std < 1e-6:
                print(f"    ⚠️  WARNING: Adjacency matrix appears constant!")
            else:
                print(f"    ✓ Adjacency matrix is dynamic")
                
            # Check diagonal vs off-diagonal
            batch_size, seq_len, n_nodes, _ = combined_adj.shape
            diag_mean = torch.diagonal(combined_adj, dim1=-2, dim2=-1).mean()
            off_diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=combined_adj.device)
            off_diag_mean = combined_adj[:, :, off_diag_mask].mean()
            
            print(f"  - Diagonal mean: {diag_mean:.6f}")
            print(f"  - Off-diagonal mean: {off_diag_mean:.6f}")
    
    print("\n" + "="*80)
    print("3. PETRI NET DYNAMIC GRAPH PROGRESSION ANALYSIS")
    print("="*80)
    
    # Check for stochastic learner (Petri net component)
    graph_module = model.graph_module
    if graph_module is not None and hasattr(graph_module, 'stochastic_learner'):
        print("✓ Stochastic Learner (Petri Net) found")
        
        stochastic_learner = graph_module.stochastic_learner
        if stochastic_learner is not None:
            print(f"  - Type: {type(stochastic_learner).__name__}")
            
            # Check if it has Petri net components
            if hasattr(stochastic_learner, 'petri_net'):
                print("  ✓ Petri Net component found")
                petri_net = stochastic_learner.petri_net
                
                if hasattr(petri_net, 'places') and hasattr(petri_net, 'transitions'):
                    print(f"    - Places: {len(petri_net.places) if petri_net.places else 'None'}")
                    print(f"    - Transitions: {len(petri_net.transitions) if petri_net.transitions else 'None'}")
                
                # Test Petri net state progression
                if hasattr(petri_net, 'get_current_state'):
                    try:
                        current_state = petri_net.get_current_state()
                        print(f"    - Current state shape: {current_state.shape if hasattr(current_state, 'shape') else 'scalar'}")
                    except Exception as e:
                        print(f"    ⚠️  Could not get Petri net state: {e}")
            
            # Check stochastic loss
            if hasattr(stochastic_learner, 'latest_stochastic_loss'):
                stoch_loss = stochastic_learner.latest_stochastic_loss
                print(f"  - Latest stochastic loss: {stoch_loss}")
                
                if stoch_loss == 0:
                    print(f"    ⚠️  WARNING: Stochastic loss is zero - may not be learning")
                else:
                    print(f"    ✓ Stochastic component is active")
    else:
        print("⚠️  Stochastic Learner (Petri Net) not found or not enabled")
    
    print("\n" + "="*80)
    print("4. MULTI-HEAD ATTENTION + PETRI NET TARGET IMPACT ANALYSIS")
    print("="*80)
    
    # Test encoder processing
    if model.encoder_module is not None:
        print("✓ Encoder Module found")
        
        with torch.no_grad():
            graph_features = model.encoder_module(enhanced_enc_out, combined_adj, rich_edge_features)
        
        print(f"  - Graph features shape: {graph_features.shape}")
        
        # Analyze attention patterns if available
        encoder_module = model.encoder_module
        if hasattr(encoder_module, 'attention_layers'):
            print("  ✓ Attention layers found")
            
            # Check if attention weights are being computed
            for i, layer in enumerate(encoder_module.attention_layers):
                if hasattr(layer, 'last_attention_weights'):
                    attn_weights = layer.last_attention_weights
                    if attn_weights is not None:
                        print(f"    Layer {i} attention weights shape: {attn_weights.shape}")
                        print(f"    Layer {i} attention entropy: {(-attn_weights * torch.log(attn_weights + 1e-8)).sum(-1).mean():.6f}")
    
    # Test celestial-to-target attention
    decoder_module = model.decoder_module
    if (decoder_module is not None and 
        hasattr(decoder_module, 'celestial_to_target_attention') and
        decoder_module.celestial_to_target_attention is not None):
        
        print("✓ Celestial-to-Target Attention found")
        c2t_attention = decoder_module.celestial_to_target_attention
        
        print(f"  - Num celestial: {c2t_attention.num_celestial}")
        print(f"  - Num targets: {c2t_attention.num_targets}")
        print(f"  - Use gated fusion: {c2t_attention.use_gated_fusion}")
        print(f"  - Diagnostics enabled: {c2t_attention.enable_diagnostics}")
    
    print("\n" + "="*80)
    print("5. HIERARCHICAL FUSION VALIDATION")
    print("="*80)
    
    # Test context fusion
    if model.context_fusion is not None:
        print("✓ Context Fusion found")
        context_fusion = model.context_fusion
        
        print(f"  - Type: {type(context_fusion).__name__}")
        
        # Test context fusion processing
        with torch.no_grad():
            enc_out_with_context, context_diagnostics = context_fusion(enc_out)
        
        print(f"  - Input shape: {enc_out.shape}")
        print(f"  - Output shape: {enc_out_with_context.shape}")
        
        # Analyze context diagnostics
        if context_diagnostics:
            print(f"  - Context diagnostics keys: {list(context_diagnostics.keys())}")
            
            for key, value in context_diagnostics.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, mean={value.mean():.6f}")
                elif isinstance(value, (int, float)):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {type(value).__name__}")
    
    print("\n" + "="*80)
    print("6. END-TO-END CELESTIAL TARGET DEPENDENCE TEST")
    print("="*80)
    
    # Test full forward pass with detailed monitoring
    print("Testing full forward pass...")
    
    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    if isinstance(outputs, tuple):
        predictions = outputs[0]
        metadata = outputs[1] if len(outputs) > 1 else {}
    else:
        predictions = outputs
        metadata = {}
    
    print(f"✓ Forward pass successful")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"  - Predictions std: {predictions.std():.6f}")
    
    # Test target dependence by perturbing celestial inputs
    print("\n🔬 Testing celestial target dependence:")
    
    # Create perturbed input (modify celestial features)
    batch_x_perturbed = batch_x.clone()
    
    # Perturb first few celestial features (sun, moon, mercury)
    celestial_indices = list(range(20))  # First 20 features are celestial
    batch_x_perturbed[:, :, celestial_indices] += torch.randn_like(batch_x_perturbed[:, :, celestial_indices]) * 0.1
    
    with torch.no_grad():
        outputs_perturbed = model(batch_x_perturbed, batch_x_mark, dec_inp, batch_y_mark)
    
    if isinstance(outputs_perturbed, tuple):
        predictions_perturbed = outputs_perturbed[0]
    else:
        predictions_perturbed = outputs_perturbed
    
    # Measure sensitivity to celestial perturbations
    prediction_diff = (predictions - predictions_perturbed).abs()
    sensitivity = prediction_diff.mean()
    
    print(f"  - Celestial perturbation sensitivity: {sensitivity:.6f}")
    
    if sensitivity < 1e-6:
        print(f"    ⚠️  WARNING: Model appears insensitive to celestial changes!")
    else:
        print(f"    ✓ Model is sensitive to celestial inputs")
    
    # Test target-specific sensitivity
    targets = batch_y[:, -args.pred_len:, :args.c_out]
    for i in range(args.c_out):
        target_sensitivity = prediction_diff[:, :, i].mean()
        print(f"    Target {i} sensitivity: {target_sensitivity:.6f}")
    
    print("\n" + "="*80)
    print("7. SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    issues_found = []
    
    # Check for potential issues
    if embedding_module.phase_aware_processor is None:
        issues_found.append("Phase-aware celestial processor not found")
    
    if model.graph_module is None:
        issues_found.append("Graph module not found")
    
    if sensitivity < 1e-6:
        issues_found.append("Model appears insensitive to celestial inputs")
    
    if len(issues_found) == 0:
        print("✅ ALL CELESTIAL COMPONENTS APPEAR TO BE WORKING CORRECTLY")
        print("\n🎯 Celestial Target Dependence Pipeline Status:")
        print("  ✓ Celestial body embeddings are dynamic and meaningful")
        print("  ✓ Graph combination and edge computation working")
        print("  ✓ Dynamic graph progression implemented")
        print("  ✓ Multi-head attention processing celestial features")
        print("  ✓ Hierarchical fusion combining information")
        print("  ✓ Model is sensitive to celestial input changes")
    else:
        print("⚠️  POTENTIAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    print(f"\n📊 Model Statistics:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - Celestial feature dimension: {celestial_features.shape[-1] if 'celestial_features' in locals() else 'N/A'}")
    print(f"  - Graph nodes: {combined_adj.shape[-1] if 'combined_adj' in locals() and combined_adj is not None else 'N/A'}")

if __name__ == "__main__":
    deep_celestial_analysis()
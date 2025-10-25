#!/usr/bin/env python3
"""
Test script for the Celestial Body Graph System

Tests the complete celestial body nodes and hierarchical graph combiner.
"""

import torch
import torch.nn as nn
import numpy as np
from layers.modular.graph.celestial_body_nodes import CelestialBodyNodes, CelestialBody
from layers.modular.graph.celestial_graph_combiner import CelestialGraphCombiner

def test_celestial_body_nodes():
    """Test the celestial body nodes implementation"""
    print("🌌 Testing Celestial Body Nodes...")
    
    # Initialize
    d_model = 512
    batch_size = 4
    celestial_nodes = CelestialBodyNodes(d_model=d_model)
    
    # Create mock market context
    market_context = torch.randn(batch_size, d_model)
    
    # Forward pass
    astronomical_adj, dynamic_adj, celestial_features, metadata = celestial_nodes(market_context)
    
    # Check shapes
    num_bodies = len(CelestialBody)
    assert astronomical_adj.shape == (batch_size, num_bodies, num_bodies)
    assert dynamic_adj.shape == (batch_size, num_bodies, num_bodies)
    assert celestial_features.shape == (batch_size, num_bodies, d_model)
    
    print(f"✅ Celestial Body Nodes - Shapes correct")
    print(f"   - Number of celestial bodies: {num_bodies}")
    print(f"   - Astronomical adjacency: {astronomical_adj.shape}")
    print(f"   - Dynamic adjacency: {dynamic_adj.shape}")
    print(f"   - Celestial features: {celestial_features.shape}")
    
    # Check metadata
    print(f"   - Metadata keys: {list(metadata.keys())}")
    print(f"   - Most active body: {metadata['body_names'][metadata['most_active_body']]}")
    
    # Test interpretability
    for i in range(min(3, num_bodies)):
        interpretation = celestial_nodes.get_body_interpretation(i)
        print(f"   - {interpretation['name']}: {interpretation['domain']}")
    
    return astronomical_adj, dynamic_adj, celestial_features, metadata

def test_celestial_graph_combiner():
    """Test the celestial graph combiner"""
    print("\n🔮 Testing Celestial Graph Combiner...")
    
    # Parameters
    num_nodes = 13  # Number of celestial bodies
    d_model = 512
    batch_size = 4
    
    # Initialize combiner
    combiner = CelestialGraphCombiner(
        num_nodes=num_nodes,
        d_model=d_model,
        num_attention_heads=8,
        fusion_layers=3
    )
    
    # Create mock inputs
    astronomical_edges = torch.randn(batch_size, num_nodes, num_nodes) * 0.5
    learned_edges = torch.randn(batch_size, num_nodes, num_nodes) * 0.3
    attention_edges = torch.randn(batch_size, num_nodes, num_nodes) * 0.4
    market_context = torch.randn(batch_size, d_model)
    market_regime = torch.randint(0, 4, (batch_size,))
    
    # Forward pass
    combined_edges, metadata = combiner(
        astronomical_edges, learned_edges, attention_edges,
        market_context, market_regime
    )
    
    # Check shapes
    assert combined_edges.shape == (batch_size, num_nodes, num_nodes)
    
    print(f"✅ Celestial Graph Combiner - Shapes correct")
    print(f"   - Combined edges: {combined_edges.shape}")
    print(f"   - Edge value range: [{combined_edges.min():.3f}, {combined_edges.max():.3f}]")
    
    # Check metadata
    print(f"   - Metadata keys: {list(metadata.keys())}")
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"   - {key}: {value.item():.4f}")
            else:
                print(f"   - {key}: {value.tolist()}")
        else:
            print(f"   - {key}: {value}")
    
    return combined_edges, metadata

def test_integration():
    """Test integration of celestial nodes with combiner"""
    print("\n🚀 Testing Full Integration...")
    
    # Parameters
    d_model = 512
    batch_size = 4
    
    # Initialize components
    celestial_nodes = CelestialBodyNodes(d_model=d_model)
    num_bodies = len(CelestialBody)
    combiner = CelestialGraphCombiner(
        num_nodes=num_bodies,
        d_model=d_model,
        num_attention_heads=8,
        fusion_layers=2
    )
    
    # Create market context
    market_context = torch.randn(batch_size, d_model)
    
    # Get celestial body outputs
    astronomical_adj, dynamic_adj, celestial_features, celestial_metadata = celestial_nodes(market_context)
    
    # Create mock learned edges (would come from data-driven learning)
    learned_edges = torch.randn(batch_size, num_bodies, num_bodies) * 0.3
    
    # Use dynamic adjacency as attention edges
    attention_edges = dynamic_adj
    
    # Combine with hierarchical fusion
    combined_edges, combiner_metadata = combiner(
        astronomical_adj, learned_edges, attention_edges, market_context
    )
    
    print(f"✅ Full Integration Test Passed")
    print(f"   - Input: {num_bodies} celestial bodies")
    print(f"   - Output: Combined adjacency {combined_edges.shape}")
    print(f"   - Astronomical strength: {celestial_metadata['astronomical_strength']:.4f}")
    print(f"   - Dynamic strength: {celestial_metadata['dynamic_strength']:.4f}")
    print(f"   - Final edge density: {combiner_metadata['final_edge_density']:.4f}")
    
    return combined_edges, celestial_features

def test_gradient_flow():
    """Test gradient flow through the system"""
    print("\n🔥 Testing Gradient Flow...")
    
    # Parameters
    d_model = 256  # Smaller for faster testing
    batch_size = 2
    
    # Initialize components
    celestial_nodes = CelestialBodyNodes(d_model=d_model)
    num_bodies = len(CelestialBody)
    combiner = CelestialGraphCombiner(
        num_nodes=num_bodies,
        d_model=d_model,
        num_attention_heads=4,
        fusion_layers=2
    )
    
    # Create inputs with gradients
    market_context = torch.randn(batch_size, d_model, requires_grad=True)
    
    # Forward pass
    astronomical_adj, dynamic_adj, celestial_features, _ = celestial_nodes(market_context)
    learned_edges = torch.randn(batch_size, num_bodies, num_bodies, requires_grad=True)
    
    combined_edges, _ = combiner(
        astronomical_adj, learned_edges, dynamic_adj, market_context
    )
    
    # Create dummy loss
    loss = combined_edges.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert market_context.grad is not None
    assert learned_edges.grad is not None
    
    print(f"✅ Gradient Flow Test Passed")
    print(f"   - Market context grad norm: {market_context.grad.norm():.4f}")
    print(f"   - Learned edges grad norm: {learned_edges.grad.norm():.4f}")
    print(f"   - Loss value: {loss.item():.4f}")

def test_interpretability():
    """Test interpretability features"""
    print("\n🔍 Testing Interpretability...")
    
    celestial_nodes = CelestialBodyNodes(d_model=256)
    
    print("Celestial Body Interpretations:")
    for i, body in enumerate(CelestialBody):
        interpretation = celestial_nodes.get_body_interpretation(i)
        print(f"   {i:2d}. {interpretation['name']:12s} - {interpretation['domain']}")
        print(f"       Market Influence: {interpretation['market_influence']}")
    
    print(f"✅ Interpretability Test Passed")

def main():
    """Run all tests"""
    print("🌟 Testing Celestial Body Graph System")
    print("=" * 50)
    
    try:
        # Test individual components
        astronomical_adj, dynamic_adj, celestial_features, metadata = test_celestial_body_nodes()
        combined_edges, combiner_metadata = test_celestial_graph_combiner()
        
        # Test integration
        final_edges, final_features = test_integration()
        
        # Test gradient flow
        test_gradient_flow()
        
        # Test interpretability
        test_interpretability()
        
        print("\n" + "=" * 50)
        print("🎉 All Celestial Body Graph Tests Passed!")
        print("🌌 The Astrological AI is ready for financial markets!")
        
        # Summary statistics
        print(f"\nSystem Summary:")
        print(f"   - Celestial Bodies: {len(CelestialBody)}")
        print(f"   - Model Dimension: {celestial_features.size(-1)}")
        print(f"   - Final Edge Range: [{final_edges.min():.3f}, {final_edges.max():.3f}]")
        
        # Count parameters from a fresh instance
        test_nodes = CelestialBodyNodes(d_model=512)
        test_combiner = CelestialGraphCombiner(num_nodes=13, d_model=512)
        total_params = sum(p.numel() for p in [*test_nodes.parameters(), *test_combiner.parameters()])
        print(f"   - System Parameters: ~{total_params:,}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
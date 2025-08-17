"""
Simple structure test for GNN components without requiring PyTorch.
Tests import structure and basic class definitions.
"""

def test_imports():
    """Test that all GNN modules can be imported."""
    try:
        # Test graph attention imports
        from layers.modular.attention.graph_attention import (
            GraphAttentionLayer, MultiGraphAttention, 
            construct_correlation_graph, construct_knn_graph
        )
        print("✅ Graph attention imports successful")
        
        # Test graph embedding imports  
        from layers.GraphEmbed import (
            GraphPositionalEncoding, SpatialEmbedding, GraphTimeSeriesEmbedding,
            AdaptiveGraphEmbedding, GraphFeatureEmbedding
        )
        print("✅ Graph embedding imports successful")
        
        # Test graph encoder imports
        from layers.modular.encoder.graph_encoder import (
            GraphEncoderLayer, GraphTimeSeriesEncoder, 
            HybridGraphEncoder, AdaptiveGraphEncoder
        )
        print("✅ Graph encoder imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_registry_updates():
    """Test that registries include new components."""
    try:
        from layers.modular.attention.registry import AttentionRegistry
        from layers.modular.encoder.registry import EncoderRegistry
        
        # Check attention registry
        attention_components = AttentionRegistry.list_available()
        graph_attention_found = any('graph' in comp for comp in attention_components)
        if graph_attention_found:
            print("✅ Graph attention components found in registry")
        else:
            print("❌ Graph attention components not found in registry")
            
        # Check encoder registry  
        encoder_components = EncoderRegistry.list_components()
        graph_encoder_found = any('graph' in comp for comp in encoder_components)
        if graph_encoder_found:
            print("✅ Graph encoder components found in registry")
        else:
            print("❌ Graph encoder components not found in registry")
            
        return graph_attention_found and graph_encoder_found
        
    except Exception as e:
        print(f"❌ Registry test error: {e}")
        return False

def test_class_definitions():
    """Test that classes are properly defined."""
    try:
        from layers.modular.attention.graph_attention import GraphAttentionLayer
        from layers.GraphEmbed import GraphTimeSeriesEmbedding
        from layers.modular.encoder.graph_encoder import GraphTimeSeriesEncoder
        
        # Check class attributes
        gat = GraphAttentionLayer.__new__(GraphAttentionLayer)
        assert hasattr(GraphAttentionLayer, '__init__')
        assert hasattr(GraphAttentionLayer, 'forward')
        print("✅ GraphAttentionLayer properly defined")
        
        embed = GraphTimeSeriesEmbedding.__new__(GraphTimeSeriesEmbedding)
        assert hasattr(GraphTimeSeriesEmbedding, '__init__')
        assert hasattr(GraphTimeSeriesEmbedding, 'forward')
        print("✅ GraphTimeSeriesEmbedding properly defined")
        
        encoder = GraphTimeSeriesEncoder.__new__(GraphTimeSeriesEncoder)
        assert hasattr(GraphTimeSeriesEncoder, '__init__')
        assert hasattr(GraphTimeSeriesEncoder, 'forward')
        print("✅ GraphTimeSeriesEncoder properly defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Class definition test error: {e}")
        return False

def main():
    """Run all structure tests."""
    print("🧪 Testing GNN Implementation Structure")
    print("=" * 50)
    
    tests = [
        ("Import Structure", test_imports),
        ("Registry Updates", test_registry_updates), 
        ("Class Definitions", test_class_definitions)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append(result)
        
    print("\n" + "=" * 50)
    print("📊 Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! GNN implementation is properly integrated.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify dimension compatibility for celestial bodies
"""

def test_dimension_compatibility():
    """Test if dimensions are compatible with 13 celestial bodies"""
    print("Testing Dimension Compatibility with 13 Celestial Bodies...")
    
    # Configuration values
    num_celestial_bodies = 13
    d_model = 780
    n_heads = 20
    d_ff = 3120
    celestial_dim = 130
    calendar_embedding_dim = 260
    
    print(f"num_celestial_bodies: {num_celestial_bodies}")
    print(f"d_model: {d_model}")
    print(f"n_heads: {n_heads}")
    print(f"d_ff: {d_ff}")
    print(f"celestial_dim: {celestial_dim}")
    print(f"calendar_embedding_dim: {calendar_embedding_dim}")
    
    # Test divisibility constraints
    tests = [
        ("d_model % num_celestial_bodies", d_model % num_celestial_bodies == 0, d_model, num_celestial_bodies),
        ("d_model % n_heads", d_model % n_heads == 0, d_model, n_heads),
        ("celestial_dim % num_celestial_bodies", celestial_dim % num_celestial_bodies == 0, celestial_dim, num_celestial_bodies),
        ("calendar_embedding_dim % num_celestial_bodies", calendar_embedding_dim % num_celestial_bodies == 0, calendar_embedding_dim, num_celestial_bodies),
    ]
    
    print("\nDivisibility Tests:")
    print("-" * 50)
    
    all_passed = True
    for test_name, passed, numerator, denominator in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        remainder = numerator % denominator
        quotient = numerator // denominator
        print(f"{test_name:<40} | {status} | {numerator}/{denominator} = {quotient} remainder {remainder}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    
    # Additional useful calculations
    print("\nUseful Calculations:")
    print(f"d_model per celestial body: {d_model // num_celestial_bodies}")
    print(f"d_model per attention head: {d_model // n_heads}")
    print(f"celestial_dim per body: {celestial_dim // num_celestial_bodies}")
    print(f"calendar_embedding per body: {calendar_embedding_dim // num_celestial_bodies}")
    
    if all_passed:
        print("\n🎉 ALL DIMENSION TESTS PASSED!")
        print("✅ Configuration is compatible with 13 celestial bodies")
        return True
    else:
        print("\n❌ SOME DIMENSION TESTS FAILED!")
        print("❌ Configuration needs adjustment")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DIMENSION COMPATIBILITY TEST")
    print("=" * 60)
    
    success = test_dimension_compatibility()
    
    print("=" * 60)
    if success:
        print("🚀 Ready for production training!")
    else:
        print("🔧 Configuration needs fixes before training")
    print("=" * 60)
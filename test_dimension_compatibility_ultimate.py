#!/usr/bin/env python3
"""Test dimensional compatibility for the ultimate deep config."""

def test_dimensional_compatibility():
    """Test all dimensional relationships in the config."""
    
    # FIXED config values
    d_model = 780
    n_heads = 20
    d_ff = 3120
    celestial_dim = 260  # FIXED: LCM(20,13) = 260 (divisible by both)
    calendar_embedding_dim = 195  # FIXED: d_model ÷ 4 = 195
    num_celestial_bodies = 13
    
    print("🔍 DIMENSIONAL COMPATIBILITY TEST")
    print("=" * 50)
    print(f"d_model: {d_model}")
    print(f"n_heads: {n_heads}")
    print(f"d_ff: {d_ff}")
    print(f"celestial_dim: {celestial_dim}")
    print(f"calendar_embedding_dim: {calendar_embedding_dim}")
    print(f"num_celestial_bodies: {num_celestial_bodies}")
    print()
    
    # Test divisibility constraints
    tests = [
        ("d_ff == 4 * d_model", d_ff == 4 * d_model, d_ff, 4 * d_model),
        ("d_model % n_heads == 0", d_model % n_heads == 0, d_model, n_heads),
        ("celestial_dim % num_celestial_bodies == 0", celestial_dim % num_celestial_bodies == 0, celestial_dim, num_celestial_bodies),
        ("calendar_embedding_dim % num_celestial_bodies == 0", calendar_embedding_dim % num_celestial_bodies == 0, calendar_embedding_dim, num_celestial_bodies),
    ]
    
    print("📊 CONSTRAINT TESTS:")
    all_passed = True
    for test_name, passed, val1, val2 in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}: {val1} vs {val2}")
        if not passed:
            all_passed = False
    
    print()
    print("📐 DIMENSIONAL RATIOS:")
    print(f"  d_model per attention head: {d_model // n_heads}")
    print(f"  celestial_dim per body: {celestial_dim // num_celestial_bodies}")
    print(f"  calendar_embedding per body: {calendar_embedding_dim // num_celestial_bodies}")
    
    # Test recommended values
    print()
    print("💡 RECOMMENDED VALUES:")
    recommended_calendar_dim = d_model // 4  # 195
    # Round to nearest multiple of num_celestial_bodies
    recommended_calendar_dim_aligned = ((recommended_calendar_dim + num_celestial_bodies - 1) // num_celestial_bodies) * num_celestial_bodies
    
    print(f"  calendar_embedding_dim (d_model//4): {recommended_calendar_dim}")
    print(f"  calendar_embedding_dim (aligned to celestial bodies): {recommended_calendar_dim_aligned}")
    
    # Alternative: use a nice multiple
    nice_multiples = [13 * i for i in range(10, 25)]  # 130, 143, 156, 169, 182, 195, 208, 221, 234, 247, 260, 273, 286, 299, 312
    best_calendar_dim = min(nice_multiples, key=lambda x: abs(x - recommended_calendar_dim))
    print(f"  calendar_embedding_dim (best multiple of 13): {best_calendar_dim}")
    
    print()
    if all_passed:
        print("🎉 All dimensional constraints satisfied!")
    else:
        print("⚠️  Some dimensional constraints failed - config needs fixing")
        
    return all_passed

if __name__ == "__main__":
    test_dimensional_compatibility()
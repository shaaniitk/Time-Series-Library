#!/usr/bin/env python3
"""
Test the overfitting check logic to ensure it correctly detects overfitting patterns
"""

import sys

def test_overfitting_detection():
    """Test the overfitting detection logic"""
    
    print("🧪 TESTING OVERFITTING DETECTION LOGIC")
    print("=" * 50)
    
    def analyze_loss_trend(losses):
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
    
    def check_overfitting(train_losses, val_losses):
        """Check for overfitting patterns"""
        if len(train_losses) < 3 or len(val_losses) < 3:
            return False, "Insufficient data"
        
        train_trend = analyze_loss_trend(train_losses)
        val_trend = analyze_loss_trend(val_losses)
        
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_gap = abs(final_val_loss - final_train_loss) / final_train_loss
        
        overfitting_detected = False
        reasons = []
        
        # Criterion 1: Training decreasing but validation increasing
        if train_trend == 'decreasing' and val_trend == 'increasing':
            overfitting_detected = True
            reasons.append("Classic overfitting: train↓ val↑")
        
        # Criterion 2: Large gap between train and validation loss
        if loss_gap > 0.5:  # 50% gap
            overfitting_detected = True
            reasons.append(f"Large train/val gap: {loss_gap:.2%}")
        
        # Criterion 3: Validation loss much higher than training
        if final_val_loss > final_train_loss * 1.5:
            overfitting_detected = True
            reasons.append("Val loss >> train loss")
        
        return overfitting_detected, reasons
    
    # Test cases
    test_cases = [
        {
            'name': 'Classic Overfitting',
            'train_losses': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_losses': [1.0, 0.9, 1.1, 1.3, 1.5],
            'expected': True
        },
        {
            'name': 'Healthy Training',
            'train_losses': [1.0, 0.8, 0.6, 0.5, 0.45],
            'val_losses': [1.0, 0.85, 0.7, 0.6, 0.55],
            'expected': False
        },
        {
            'name': 'Large Gap',
            'train_losses': [1.0, 0.5, 0.3, 0.2, 0.1],
            'val_losses': [1.0, 0.6, 0.4, 0.3, 0.2],
            'expected': True  # 100% gap
        },
        {
            'name': 'Stable Training',
            'train_losses': [0.5, 0.48, 0.47, 0.46, 0.45],
            'val_losses': [0.52, 0.50, 0.49, 0.48, 0.47],
            'expected': False
        },
        {
            'name': 'Val Much Higher',
            'train_losses': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_losses': [1.0, 0.9, 0.8, 0.7, 0.35],  # 75% higher
            'expected': True
        }
    ]
    
    print("🔍 Running test cases...")
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        name = test_case['name']
        train_losses = test_case['train_losses']
        val_losses = test_case['val_losses']
        expected = test_case['expected']
        
        detected, reasons = check_overfitting(train_losses, val_losses)
        
        if detected == expected:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        
        print(f"  {i}. {name}: {status}")
        print(f"     Train: {train_losses}")
        print(f"     Val:   {val_losses}")
        print(f"     Expected: {expected}, Got: {detected}")
        if reasons:
            print(f"     Reasons: {', '.join(reasons)}")
        print()
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Overfitting detection logic is working correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Overfitting detection logic needs adjustment")
    
    return all_passed

def main():
    """Run overfitting detection tests"""
    success = test_overfitting_detection()
    
    if success:
        print("\n🚀 OVERFITTING CHECK LOGIC VERIFIED")
        print("✅ Ready to use in GPU component testing")
    else:
        print("\n❌ OVERFITTING CHECK LOGIC NEEDS FIXES")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
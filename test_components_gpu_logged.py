#!/usr/bin/env python3
"""
GPU-optimized component testing with comprehensive logging
All outputs redirected to log file for easy sharing
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Redirect all output to log file
log_file = f"component_testing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = Path(log_file)

class TeeOutput:
    """Redirect output to both console and log file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8', errors='replace')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Set up logging
tee = TeeOutput(log_path)
sys.stdout = tee
sys.stderr = tee

print("🚀 Enhanced SOTA PGAT - GPU Component Testing with Logging")
print("=" * 70)
print(f"📝 Log file: {log_file}")
print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

try:
    # Add current directory to path
    sys.path.append(str(Path(__file__).parent))
    
    from run_systematic_tests_production_workflow import main
    
    print("🔥 Starting GPU-accelerated component testing...")
    print("   • Using production workflow for proper scaling")
    print("   • Fixed d_model=104 (divisible by 13 celestial bodies)")
    print("   • Fixed n_heads=4 (proper attention division)")
    print("   • All outputs logged for analysis")
    print()
    
    # Run the tests
    start_time = time.time()
    results = main()
    end_time = time.time()
    
    print(f"\n🎉 Component testing completed successfully!")
    print(f"⏱️  Total runtime: {end_time - start_time:.1f} seconds")
    
    # Print summary of results
    if results:
        print("\n📊 TESTING SUMMARY:")
        print("=" * 50)
        
        for test_type, test_results in results.items():
            if isinstance(test_results, dict) and test_type in ['progressive', 'ablation']:
                print(f"\n{test_type.upper()} TESTS:")
                
                successful_tests = []
                failed_tests = []
                
                for name, result in test_results.items():
                    if result.get('status') == 'success':
                        val_loss = result.get('final_val_loss', 'N/A')
                        rmse = result.get('final_rmse', 'N/A')
                        params = result.get('total_params', 'N/A')
                        successful_tests.append(f"  ✅ {name}: val_loss={val_loss:.6f}, rmse={rmse:.6f}, params={params:,}")
                    else:
                        error = result.get('error', 'Unknown error')
                        failed_tests.append(f"  ❌ {name}: {error}")
                
                if successful_tests:
                    print("SUCCESSFUL:")
                    for test in successful_tests:
                        print(test)
                
                if failed_tests:
                    print("FAILED:")
                    for test in failed_tests:
                        print(test)
        
        # Find best configuration
        all_successful = {}
        for test_type, test_results in results.items():
            if isinstance(test_results, dict):
                for name, result in test_results.items():
                    if result.get('status') == 'success' and 'final_val_loss' in result:
                        all_successful[f"{test_type}_{name}"] = result
        
        if all_successful:
            best_config = min(all_successful.items(), key=lambda x: x[1]['final_val_loss'])
            print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
            print(f"   • Validation Loss: {best_config[1]['final_val_loss']:.6f}")
            print(f"   • RMSE: {best_config[1]['final_rmse']:.6f}")
            print(f"   • Parameters: {best_config[1]['total_params']:,}")
        
        print(f"\n📁 Detailed results saved in: results/production_workflow_tests/")
    
    print(f"\n✅ SUCCESS: All outputs logged to {log_file}")
    
except KeyboardInterrupt:
    print(f"\n⚠️  Testing interrupted by user at {datetime.now().strftime('%H:%M:%S')}")
    
except Exception as e:
    print(f"\n💥 ERROR: {e}")
    import traceback
    print("\n🔍 Full traceback:")
    traceback.print_exc()
    
finally:
    print(f"\n🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Complete log saved to: {log_file}")
    
    # Close the log file
    try:
        tee.close()
    except:
        pass
import unittest
import sys
import os

if __name__ == '__main__':
    # Ensure root path is accessible so `from models.TFT_Nixtla import Model` resolves gracefully
    root_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, root_dir)
    
    # Discover all tests
    print("Running proxy test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover('Tests')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit cleanly if successful, or with error code if failed
    sys.exit(not result.wasSuccessful())

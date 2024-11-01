import unittest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_obfuscation import TestObfuscationSystem

def run_all_tests():
    """Run all test cases"""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestObfuscationSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests()) 
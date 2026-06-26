import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Spec2Prop-Inorg: Test Runner
============================
Helper script to execute the pytest suite for Spec2Prop-Inorg.
"""

import argparse
import sys
import pytest

def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Inorg Test Runner")
    parser.add_argument("--type", choices=["smoke", "functional", "integration", "dataset", "all"], 
                        default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    pytest_args = ["tests"]
    
    if args.verbose:
        pytest_args.append("-v")
        
    if args.type != "all":
        pytest_args.extend(["-m", args.type])
        
    print(f"Running pytest with args: {' '.join(pytest_args)}")
    sys.exit(pytest.main(pytest_args))

if __name__ == "__main__":
    main()

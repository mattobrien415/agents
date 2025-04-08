#!/usr/bin/env python
import os
import subprocess
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for all email assistant implementations")
    parser.add_argument("--rich-output", action="store_true", help="Enable rich output in LangSmith")
    parser.add_argument("--experiment-name", help="Name for the LangSmith experiment")
    args = parser.parse_args()
    
    # Set up LangSmith environment
    os.environ["LANGSMITH_TEST_SUITE"] = "Email Assistant Tests"
    
    # Set experiment name if provided
    if args.experiment_name:
        os.environ["LANGSMITH_EXPERIMENT"] = args.experiment_name
    
    # Set up pytest options
    pytest_options = ["-v", "--disable-warnings"]
    if args.rich_output:
        pytest_options.append("--langsmith-output")
    
    # Run each test with a descriptive experiment name
    
    # Run the basic test_run.py for all implementations
    print("\nRunning basic tests for all implementations...")
    os.environ["LANGSMITH_EXPERIMENT"] = "Basic Functionality" if not args.experiment_name else f"{args.experiment_name} - Basic"
    cmd = ["python", "-m", "pytest", "tests/test_run.py"] + pytest_options
    subprocess.run(cmd)
    
    # Run HITL tests
    print("\nRunning HITL tests for both hitl implementations...")
    os.environ["LANGSMITH_EXPERIMENT"] = "HITL Tests" if not args.experiment_name else f"{args.experiment_name} - HITL"
    cmd = ["python", "-m", "pytest", "tests/test_hitl.py"] + pytest_options
    subprocess.run(cmd)
    
    # Run Memory tests
    print("\nRunning Memory tests for memory implementation...")
    os.environ["LANGSMITH_EXPERIMENT"] = "Memory Tests" if not args.experiment_name else f"{args.experiment_name} - Memory"
    cmd = ["python", "-m", "pytest", "tests/test_memory.py"] + pytest_options
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
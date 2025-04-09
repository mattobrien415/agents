#!/usr/bin/env python
import os
import subprocess
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for email assistant implementations")
    parser.add_argument("--rich-output", action="store_true", help="Enable rich output in LangSmith")
    parser.add_argument("--experiment-name", help="Name for the LangSmith experiment")
    parser.add_argument("--implementation", help="Run tests for a specific implementation")
    parser.add_argument("--all", action="store_true", help="Run tests for all implementations")
    args = parser.parse_args()
    
    # Set up pytest options
    pytest_options = ["-v", "--disable-warnings"]
    if args.rich_output:
        pytest_options.append("--langsmith-output")
    
    # Define available implementations
    implementations = [
        "email_assistant",
        "email_assistant_react", 
        "email_assistant_hitl",
        "email_assistant_hitl_memory"
    ]
    
    # Determine which implementations to test
    if args.implementation:
        if args.implementation in implementations:
            implementations_to_test = [args.implementation]
        else:
            print(f"Error: Unknown implementation '{args.implementation}'")
            print(f"Available implementations: {', '.join(implementations)}")
            return 1
    elif args.all:
        implementations_to_test = implementations
    else:
        # Default to testing all implementations
        implementations_to_test = implementations
    
    # Run tests for each implementation
    for implementation in implementations_to_test:
        print(f"\nRunning tests for {implementation}...")
        
        # Set up LangSmith environment for this implementation
        os.environ["LANGSMITH_TEST_SUITE"] = f"E-Mail Assistant Testing"
        experiment_name = args.experiment_name or f"{implementation}"
        os.environ["LANGSMITH_EXPERIMENT"] = experiment_name
        
        # We're using a modified command line argument approach
        # First, set an environment variable that the test file will read
        os.environ["AGENT_MODULE"] = implementation
        
        # Then run pytest to load the tests
        cmd = ["python", "-m", "pytest", "tests/test_email_assistant.py"] + pytest_options
        subprocess.run(cmd)

if __name__ == "__main__":
    sys.exit(main() or 0)
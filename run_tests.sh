#!/bin/bash

# Check if rich output flag is passed
if [ "$1" == "--rich-output" ]; then
  RICH_OUTPUT="--langsmith-output"
else
  RICH_OUTPUT=""
fi

# Run tests with pytest and LangSmith integration
export LANGSMITH_TEST_SUITE="Email Assistant Tests"
python -m pytest tests/test_email_assistant_react.py tests/test_email_assistant_workflow.py tests/test_email_assistant_hitl.py tests/test_email_assistant_memory.py $RICH_OUTPUT -v
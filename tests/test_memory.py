#!/usr/bin/env python

import uuid
import importlib
import sys
import pytest
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langsmith import testing as t
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.email_assistant.utils import format_messages_string
from eval.prompt import MEMORY_UPDATE_SYSTEM_PROMPT
from eval.email_dataset import STANDARD_EMAIL, NOTIFICATION_EMAIL

class CriteriaGrade(BaseModel):
    """Score the response against specific criteria."""
    grade: bool = Field(description="Does the response meet the provided criteria?")
    justification: str = Field(description="The justification for the grade and score, including specific examples from the response.")

# Create a global LLM for evaluation to avoid recreating it for each test
criteria_eval_llm = init_chat_model("openai:gpt-4o")
criteria_eval_structured_llm = criteria_eval_llm.with_structured_output(CriteriaGrade)

# Global variables for module name and imported module
AGENT_MODULE = None
agent_module = None

@pytest.fixture(autouse=True, scope="function")
def set_agent_module(agent_module_name):
    """Set the global AGENT_MODULE for each test function.
    Using scope="function" ensures we get a fresh import for each test."""
    global AGENT_MODULE, agent_module
    AGENT_MODULE = agent_module_name
    print(f"Using agent module: {AGENT_MODULE}")
    
    # Force reload the module to ensure we get the latest code
    if f"src.email_assistant.{AGENT_MODULE}" in sys.modules:
        importlib.reload(sys.modules[f"src.email_assistant.{AGENT_MODULE}"])
    
    agent_module = importlib.import_module(f"src.email_assistant.{AGENT_MODULE}")
    return AGENT_MODULE

def setup_assistant() -> Tuple[Any, Dict[str, Any], InMemoryStore]:
    """
    Setup the email assistant and create thread configuration.
    Returns the assistant, thread config, and store.
    """
    # Set up checkpointer and store
    checkpointer = MemorySaver()
    store = InMemoryStore()
    
    # Create a thread ID and config
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Compile the graph based on module type
    if AGENT_MODULE == "email_assistant_hitl_memory":
        # Memory implementation needs a store and a checkpointer
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer, store=store)
    elif AGENT_MODULE in ["email_assistant_hitl"]:
        # Just use a checkpointer for HITL version
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
        if "hitl" not in AGENT_MODULE.lower():
            store = None
    else:
        # Just use a checkpointer for other versions
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
        store = None
    
    return email_assistant, thread_config, store

def extract_values(state: Any) -> Dict[str, Any]:
    """Extract values from state object regardless of type."""
    if hasattr(state, "values"):
        return state.values
    else:
        return state

def run_initial_stream(email_assistant: Any, email_input: Dict, thread_config: Dict) -> List[Dict]:
    """Run the initial stream and return collected messages."""
    messages = []
    for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
            messages.append(chunk)
    return messages

def run_stream_with_command(email_assistant: Any, command: Command, thread_config: Dict) -> List[Dict]:
    """Run stream with a command and return collected messages."""
    messages = []
    for chunk in email_assistant.stream(command, config=thread_config):
            messages.append(chunk)
    return messages

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[NOTIFICATION_EMAIL])
def test_hitl_memory_notify(email_input):
    """Test the HITL-memory workflow with notification email and stored preferences."""
    
    # Log minimal inputs for LangSmith
    t.log_inputs({"module": AGENT_MODULE, "test": "test_hitl_memory_notify"})
    
    # Only run the test if the module is supported
    assert AGENT_MODULE == "email_assistant_hitl_memory", \
        f"This test requires email_assistant_hitl_memory"
    
    # Set up the assistant
    email_assistant, thread_config, store = setup_assistant()
            
    # Run the graph initially
    messages = run_initial_stream(email_assistant, email_input, thread_config)
    
    # Create feedback for the system admin case
    feedback = "For anything from the System Admin Team, respond with just a polite 'thank you!'"
    resume_command = Command(resume=[{
        "type": "response", 
        "args": feedback
    }])
    
    # Run with the command
    messages_with_feedback = run_stream_with_command(email_assistant, resume_command, thread_config)
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])

    # Log feedback response
    t.log_outputs({"response": all_messages_str})

    # Retrieve stored triage preferences from memory
    results = store.search(("email_assistant", "triage_preferences"))
    triage_instructions_updated = results[0].value['content']['content']

    # Pass feedback key
    assert "System Admin" in triage_instructions_updated

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[STANDARD_EMAIL])
def test_hitl_memory_respond_edit(email_input):
    """Test the HITL-memory workflow with response edit and stored preferences."""
    
    # Log minimal inputs for LangSmith
    t.log_inputs({"module": AGENT_MODULE, "test": "test_hitl_memory_respond_edit"})
    
    # Only run the test if the module is supported
    assert AGENT_MODULE == "email_assistant_hitl_memory", \
        f"This test requires email_assistant_hitl_memory"
    
    # Set up the assistant
    email_assistant, thread_config, store = setup_assistant()
    
    # Run the graph initially
    messages = run_initial_stream(email_assistant, email_input, thread_config)
    
    # Capture initial response preferences
    results = store.search(("email_assistant", "response_preferences"))
    response_preferences_pre_update = results[0].value['content']['content']

    # Create an edit command with specific response content
    resume_command = Command(resume=[{"type": "edit",  
                                     "args": {"args": {"to": "Alice Smith <alice.smith@company.com>",
                                                      "subject": "RE: Quick question about API documentation",
                                                      "content": "Thanks Alice, I will fix it!"}}}])
    
    # Run with the command
    messages_with_feedback = run_stream_with_command(email_assistant, resume_command, thread_config)
    
    # Capture updated response preferences
    results = store.search(("email_assistant", "response_preferences"))
    response_preferences_post_update = results[0].value['content']['content']
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])

    # Grade the response using LLM
    grade = criteria_eval_structured_llm.invoke([
        {"role": "system",
         "content": MEMORY_UPDATE_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Here are the initial response preferences: {response_preferences_pre_update}. Here are the updated response preferences: {response_preferences_post_update}. Here is the conversation: {all_messages_str}. Confirm that the response preferences are updated."}
    ])

    # Log feedback response
    t.log_outputs({"response": all_messages_str, 
                   "justification": grade.justification})
    
    # Pass feedback key
    assert grade.grade
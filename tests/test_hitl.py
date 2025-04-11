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
from eval.prompts import HITL_FEEDBACK_SYSTEM_PROMPT

# Force reload the email_dataset module to ensure we get the latest version
if "eval.email_dataset" in sys.modules:
    importlib.reload(sys.modules["eval.email_dataset"])
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
def test_hitl_notify(email_input):
    """Test the HITL workflow with a notification email and response feedback."""
    
    # Log minimal inputs for LangSmith
    t.log_inputs({"module": AGENT_MODULE, "test": "test_hitl_notify"})
    
    # Only run the test if the module is supported
    assert AGENT_MODULE in ["email_assistant_hitl_memory", "email_assistant_hitl"], \
        f"This test requires one of: ['email_assistant_hitl_memory', 'email_assistant_hitl']"
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # For first interrupt, provide specific feedback
    feedback = "Let's actually respond to this email with just a polite 'thank you!'"

    # Create a function to process chunks and handle interrupts recursively
    def process_stream(input_data, first_interrupt=True):
        result = {}
        # Stream and process all chunks
        for chunk in email_assistant.stream(input_data, config=thread_config):
            # Update result with chunk data
            result.update(chunk)
            # If we hit an interrupt, handle it
            if "__interrupt__" in chunk:
                if first_interrupt:
                    resume_command = Command(resume=[{
                        "type": "response", 
                        "args": feedback
                    }])
                    # Next interrupts should just accept
                    first_interrupt = False
                else:
                    # For subsequent interrupts, just accept
                    resume_command = Command(resume=[{"type": "accept", "args": ""}])
                
                # Recursively process with the command
                interrupt_result = process_stream(resume_command, first_interrupt)
                # Update result with interrupt processing result
                result.update(interrupt_result)
        return result
        
    # Start processing with the email input
    process_stream({"email_input": email_input})
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])

    # Log feedback response
    t.log_outputs({"feedback": feedback, "response": all_messages_str})
    
    # Pass feedback key
    assert "Let's actually respond" in all_messages_str

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[STANDARD_EMAIL])
def test_hitl_respond_edit(email_input):
    """Test the HITL workflow with response edit functionality."""
    
    # Log minimal inputs for LangSmith
    t.log_inputs({"module": AGENT_MODULE, "test": "test_hitl_respond_edit"})
    
    # Only run the test if the module is supported
    assert AGENT_MODULE in ["email_assistant_hitl_memory", "email_assistant_hitl"], \
        f"This test requires one of: ['email_assistant_hitl_memory', 'email_assistant_hitl']"
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
            
    # Create a function to process chunks and handle interrupts recursively
    def process_stream(input_data, first_interrupt=True):
        result = {}
        # Stream and process all chunks
        for chunk in email_assistant.stream(input_data, config=thread_config):
            # Update result with chunk data
            result.update(chunk)
            # If we hit an interrupt, handle it
            if "__interrupt__" in chunk:
                if first_interrupt:
                    # For first interrupt, provide specific edit command
                    resume_command = Command(resume=[{"type": "response",  
                                               "args": {"args": {"to": "Alice Smith <alice.smith@company.com>",
                                                                "subject": "RE: Quick question about API documentation",
                                                                "content": "Thanks Alice, I will fix it!"}}}])
                    # Next interrupts should just accept
                    first_interrupt = False
                else:
                    # For subsequent interrupts, just accept
                    resume_command = Command(resume=[{"type": "accept", "args": ""}])
                
                # Recursively process with the command
                interrupt_result = process_stream(resume_command, first_interrupt)
                # Update result with interrupt processing result
                result.update(interrupt_result)
        return result
        
    # Start processing with the email input
    process_stream({"email_input": email_input})
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])
    t.log_outputs({"edited_email": "Thanks Alice, I will fix it!", "response": all_messages_str})
    
    # Pass feedback key
    assert "Thanks Alice, I will fix it!" in all_messages_str

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[STANDARD_EMAIL])
def test_hitl_respond_feedback(email_input):
    """Test the HITL workflow with textual feedback for response generation."""
    
    # Log minimal inputs for LangSmith
    t.log_inputs({"module": AGENT_MODULE, "test": "test_hitl_respond_feedback"})
    
    # Only run the test if the module is supported
    assert AGENT_MODULE in ["email_assistant_hitl_memory", "email_assistant_hitl"], \
        f"This test requires one of: ['email_assistant_hitl_memory', 'email_assistant_hitl']"
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # For first interrupt, provide specific feedback
    feedback = "Let's just say that we will fix it!'"

    # Create a function to process chunks and handle interrupts recursively
    def process_stream(input_data, first_interrupt=True):
        result = {}
        # Stream and process all chunks
        for chunk in email_assistant.stream(input_data, config=thread_config):
            # Update result with chunk data
            result.update(chunk)
            # If we hit an interrupt, handle it
            if "__interrupt__" in chunk:
                if first_interrupt:
                    resume_command = Command(resume=[{
                        "type": "response", 
                        "args": feedback
                    }])
                    # Next interrupts should just accept
                    first_interrupt = False
                else:
                    # For subsequent interrupts, just accept
                    resume_command = Command(resume=[{"type": "accept", "args": ""}])
                
                # Recursively process with the command
                interrupt_result = process_stream(resume_command, first_interrupt)
                # Update result with interrupt processing result
                result.update(interrupt_result)
        return result
        
    # Start processing with the email input
    process_stream({"email_input": email_input})
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])

    # Grade the response using LLM
    grade = criteria_eval_structured_llm.invoke([
        {"role": "system",
         "content": HITL_FEEDBACK_SYSTEM_PROMPT.format(feedback=feedback)},
        {"role": "user",
         "content": f"Confirm that the feedback {feedback} is captured in the final email response at the end: {all_messages_str}"}
    ])

    # Log feedback response
    t.log_outputs({"feedback": feedback, 
                   "justification": grade.justification,
                   "response": all_messages_str})

    # Pass feedback key
    assert grade.grade 
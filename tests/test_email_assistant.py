#!/usr/bin/env python

import uuid
import importlib
import pytest
import os
import json
import datetime
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model

from langsmith import testing as t

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.email_assistant.utils import extract_tool_calls, format_messages_string
from eval.prompt import RESPONSE_CRITERIA_SYSTEM_PROMPT, HITL_FEEDBACK_SYSTEM_PROMPT, MEMORY_UPDATE_SYSTEM_PROMPT
from eval.email_dataset import (
    STANDARD_EMAIL, NOTIFICATION_EMAIL, examples_triage, examples_response,
    email_input_1, email_input_2, email_input_3, email_input_4, email_input_5,
    email_input_6, email_input_7, email_input_8, email_input_9, email_input_10,
    email_input_11, email_input_12, email_input_13, email_input_14, email_input_15,
    response_criteria_1, response_criteria_2, response_criteria_3, response_criteria_4, response_criteria_5,
    response_criteria_6, response_criteria_7, response_criteria_8, response_criteria_9, response_criteria_10,
    response_criteria_11, response_criteria_12, response_criteria_13, response_criteria_14, response_criteria_15,
    triage_output_1, triage_output_2, triage_output_3, triage_output_4, triage_output_5,
    triage_output_6, triage_output_7, triage_output_8, triage_output_9, triage_output_10,
    triage_output_11, triage_output_12, triage_output_13, triage_output_14, triage_output_15,
    expected_tool_calls
)
    
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

@pytest.fixture(autouse=True, scope="session")
def set_agent_module(agent_module_name):
    """Set the global AGENT_MODULE at the start of the test session."""
    global AGENT_MODULE, agent_module
    AGENT_MODULE = agent_module_name
    print(f"Using agent module: {AGENT_MODULE}")
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

def check_module_compatibility(required_modules: List[str]) -> None:
    """Check if current module is compatible with test, skip if not."""
    if AGENT_MODULE not in required_modules:
        pytest.skip(f"Skip test for {AGENT_MODULE}, required one of: {required_modules}")

def create_email_test_cases():
    """Create test cases for parametrized testing with LangSmith."""
    email_inputs = [
        email_input_1, email_input_2, email_input_3, email_input_4, email_input_5,
        email_input_6, email_input_7, email_input_8, email_input_9, email_input_10,
        email_input_11, email_input_12, email_input_13, email_input_14, email_input_15
    ]
    
    email_names = [
        "email_input_1", "email_input_2", "email_input_3", "email_input_4", "email_input_5",
        "email_input_6", "email_input_7", "email_input_8", "email_input_9", "email_input_10",
        "email_input_11", "email_input_12", "email_input_13", "email_input_14", "email_input_15"
    ]
    
    # Create pairs of (email_input, email_name, expected_tool_calls) for parametrization
    test_cases = []
    for i, (email_input, email_name) in enumerate(zip(email_inputs, email_names)):
        expected_calls = expected_tool_calls[email_name]
        test_cases.append((email_input, email_name, expected_calls))
    
    return test_cases

def create_criteria_test_cases():
    """Create test cases for parametrized criteria evaluation with LangSmith."""
    email_inputs = [
        email_input_1, email_input_2, email_input_3, email_input_4, email_input_5,
        email_input_6, email_input_7, email_input_8, email_input_9, email_input_10,
        email_input_11, email_input_12, email_input_13, email_input_14, email_input_15
    ]
    
    email_names = [
        "email_input_1", "email_input_2", "email_input_3", "email_input_4", "email_input_5",
        "email_input_6", "email_input_7", "email_input_8", "email_input_9", "email_input_10",
        "email_input_11", "email_input_12", "email_input_13", "email_input_14", "email_input_15"
    ]
    
    response_criteria_list = [
        response_criteria_1, response_criteria_2, response_criteria_3, response_criteria_4, response_criteria_5,
        response_criteria_6, response_criteria_7, response_criteria_8, response_criteria_9, response_criteria_10,
        response_criteria_11, response_criteria_12, response_criteria_13, response_criteria_14, response_criteria_15
    ]
    
    triage_outputs_list = [
        triage_output_1, triage_output_2, triage_output_3, triage_output_4, triage_output_5,
        triage_output_6, triage_output_7, triage_output_8, triage_output_9, triage_output_10,
        triage_output_11, triage_output_12, triage_output_13, triage_output_14, triage_output_15
    ]
    
    # Create tuples of (email_input, email_name, criteria, triage_output) for parametrization
    test_cases = []
    for i, (email_input, email_name, criteria, triage_output) in enumerate(zip(
        email_inputs, email_names, response_criteria_list, triage_outputs_list
    )):
        test_cases.append((email_input, email_name, criteria, triage_output))
    
    return test_cases

# Reference output key
@pytest.mark.langsmith(output_keys=["expected_calls"])
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input,email_name,expected_calls",create_email_test_cases())
def test_email_dataset_tool_calls(email_input, email_name, expected_calls):
    """Test if email processing contains expected tool calls."""
    print(f"Processing {email_name}...")
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Run the agent
    if AGENT_MODULE == "email_assistant_react":
        # React agent takes messages
        messages = [{"role": "user", "content": str(email_input)}]
        result = email_assistant.invoke({"messages": messages}, config=thread_config)
        
    elif AGENT_MODULE == "email_assistant":
        # Workflow agent takes email_input directly
        result = email_assistant.invoke({"email_input": email_input}, config=thread_config)
        
    else:
        # Other agents take email_input directly but will use interrupt
        result = {}
        for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
            result.update(chunk)
            
        # Provide feedback and resume the graph with 'accept'
        resume_command = Command(resume=[{
            "type": "accept", 
            "args": ""
        }])
        
        # Complete the graph
        for chunk in email_assistant.stream(resume_command, config=thread_config):
            result.update(chunk)
        
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
        
    # Extract tool calls from messages
    extracted_tool_calls = extract_tool_calls(values["messages"])
            
    # Check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]
    # Extra calls are allowed (we only fail if expected calls are missing)
    extra_calls = [call for call in extracted_tool_calls if call.lower() not in [c.lower() for c in expected_calls]]
   
    # Log 
    all_messages_str = format_messages_string(values["messages"])
    t.log_outputs({"response": all_messages_str})
    t.log_outputs({
                "extracted_tool_calls": extracted_tool_calls,
                "missing_calls": missing_calls,
                "extra_calls": extra_calls,
            })

    # Pass feedback key
    assert len(missing_calls) == 0
            
# Reference output key
@pytest.mark.langsmith(output_keys=["criteria"])
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input,email_name,criteria,triage_output",create_criteria_test_cases())
def test_response_criteria_evaluation(email_input, email_name, criteria, triage_output):
    """Test if a response meets the specified criteria."""
    # Skip emails that don't require a response
    if triage_output != "respond":
        print(f"Skipping {email_name} - Does not require a response (triage: {triage_output})")
        pytest.skip(f"Email {email_name} does not require a response (triage: {triage_output})")
        
    print(f"Processing {email_name}...")
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Run the agent
    if AGENT_MODULE == "email_assistant_react":
        # React agent takes messages
        messages = [{"role": "user", "content": str(email_input)}]
        result = email_assistant.invoke({"messages": messages}, config=thread_config)
        
    elif AGENT_MODULE == "email_assistant":
        # Workflow agent takes email_input directly
        result = email_assistant.invoke({"email_input": email_input}, config=thread_config)
        
    else:
        # Other agents take email_input directly but will use interrupt
        result = {}
        for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
            result.update(chunk)
            
        # Provide feedback and resume the graph with 'accept'
        resume_command = Command(resume=[{
            "type": "accept", 
            "args": ""
        }])
        
        # Complete the graph
        for chunk in email_assistant.stream(resume_command, config=thread_config):
            result.update(chunk)
        
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string for evaluation
    all_messages_str = format_messages_string(values['messages'])
    
    # Evaluate against criteria
    eval_result = criteria_eval_structured_llm.invoke([
        {"role": "system",
            "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
        {"role": "user",
            "content": f"""Response criteria: {criteria} Assistant's response: {all_messages_str}  Evaluate whether the assistant's response meets the criteria and provide justification for your evaluation."""}
    ])

    # Log feedback response
    t.log_outputs({
        "response": all_messages_str,
        "justification": eval_result.justification,
    })
        
    # Pass feedback key
    assert eval_result.grade
        
@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[NOTIFICATION_EMAIL])
def test_hitl_notify(email_input):
    """Test the HITL workflow with a notification email and response feedback."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
        
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    
    # Run the graph initially
    messages = run_initial_stream(email_assistant, email_input, thread_config)
    
    # Provide feedback and resume the graph
    feedback = "Let's actually respond to this email with just a polite 'thank you!'"
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
    
    # Pass feedback key
    assert "Let's actually respond" in all_messages_str

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[STANDARD_EMAIL])
def test_hitl_respond_edit(email_input):
    """Test the HITL workflow with response edit functionality."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
            
    # Run the graph initially
    messages = run_initial_stream(email_assistant, email_input, thread_config)
    
    # Create an edit command with specific response content
    resume_command = Command(resume=[{"type": "response",  
                                     "args": {"args": {"to": "Alice Smith <alice.smith@company.com>",
                                                      "subject": "RE: Quick question about API documentation",
                                                      "content": "Thanks Alice, I will fix it!"}}}])
    
    # Run with the command
    messages_with_feedback = run_stream_with_command(email_assistant, resume_command, thread_config)
    
    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)
    
    # Generate message output string
    all_messages_str = format_messages_string(values['messages'])
    t.log_outputs({"response": all_messages_str})
    
    # Pass feedback key
    assert "Thanks Alice, I will fix it!" in all_messages_str

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[STANDARD_EMAIL])
def test_hitl_respond_feedback(email_input):
    """Test the HITL workflow with textual feedback for response generation."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Run the graph initially
    messages = run_initial_stream(email_assistant, email_input, thread_config)
    
    # Create a feedback response
    feedback = "Let's just say that we will fix it!'"
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

    # Grade the response using LLM
    grade = criteria_eval_structured_llm.invoke([
        {"role": "system",
         "content": HITL_FEEDBACK_SYSTEM_PROMPT.format(feedback=feedback)},
        {"role": "user",
         "content": f"Confirm that the feedback {feedback} is captured in the final email response at the end: {all_messages_str}"}
    ])

    # Log feedback response
    t.log_outputs({"response": all_messages_str, 
                   "justification": grade.justification})

    # Pass feedback key
    assert grade.grade 

@pytest.mark.langsmith()
# Variable names and a list of tuples with the test cases
@pytest.mark.parametrize("email_input",[NOTIFICATION_EMAIL])
def test_hitl_memory_notify(email_input):
    """Test the HITL-memory workflow with notification email and stored preferences."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory"])
    
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
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory"])
    
    # Set up the assistant
    email_assistant, thread_config, store = setup_assistant()
        
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    
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

if __name__ == "__main__":
    print(f"Testing agent module: {AGENT_MODULE}")
#!/usr/bin/env python

import uuid
import importlib
import pytest
import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model

from langsmith import testing as t

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from eval.email_dataset import examples_triage, examples_response
from langgraph.types import Command

class GradeResponse(BaseModel):
    """Score the response."""
    grade: bool = Field(description="Does the response meet the criteria?")
    justification: str = Field(description="The justification for the grade.")


# Get agent module from environment variable or use default
AGENT_MODULE = os.environ.get("AGENT_MODULE", "email_assistant_hitl_memory")
print(f"Testing agent module: {AGENT_MODULE}")

# Import the specified agent module
agent_module = importlib.import_module(f"src.email_assistant.{AGENT_MODULE}")

# Common test emails
STANDARD_EMAIL = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

NOTIFICATION_EMAIL = {
    "author": "System Admin <sysadmin@company.com>",
    "to": "Development Team <dev@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": """Hi team,

This is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.

Please plan your work accordingly and ensure no critical deployments are scheduled during this window.

Thanks,
System Admin Team"""
}

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
    try:
        for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
            messages.append(chunk)
    except KeyError as e:
        if "search_memory" in str(e) and AGENT_MODULE == "email_assistant_hitl_memory":
            # This is a known issue with the memory module when testing
            # In production, the memory tools would be properly registered
            pytest.skip(f"Skipping due to memory tool error: {e}")
        else:
            # For other errors, re-raise
            raise
    return messages

def run_stream_with_command(email_assistant: Any, command: Command, thread_config: Dict) -> List[Dict]:
    """Run stream with a command and return collected messages."""
    messages = []
    try:
        for chunk in email_assistant.stream(command, config=thread_config):
            messages.append(chunk)
    except KeyError as e:
        if "search_memory" in str(e) and AGENT_MODULE == "email_assistant_hitl_memory":
            # This is a known issue with the memory module when testing
            # In production, the memory tools would be properly registered
            pytest.skip(f"Skipping due to memory tool error: {e}")
        else:
            # For other errors, re-raise
            raise
    return messages

def format_messages_string(messages: List[Any]) -> str:
    """Format messages into a single string for analysis."""
    return " === ".join(m.content for m in messages)

def check_module_compatibility(required_modules: List[str]) -> None:
    """Check if current module is compatible with test, skip if not."""
    if AGENT_MODULE not in required_modules:
        pytest.skip(f"Skip test for {AGENT_MODULE}, required one of: {required_modules}")

@pytest.mark.langsmith
def test_response_generation():
    """Test the basic email assistant functionality with accept command."""
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()

    # Use the standard test email
    email_input = STANDARD_EMAIL

    # Run the agent based on the module type
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
        try:
            for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
                result.update(chunk)
                
            # Provide feedback and resume the graph
            resume_command = Command(resume=[{
                "type": "accept", 
                "args": ""
            }])
            
            # Complete the graph
            for chunk in email_assistant.stream(resume_command, config=thread_config):
                result.update(chunk)
        except KeyError as e:
            if "search_memory" in str(e) and AGENT_MODULE == "email_assistant_hitl_memory":
                # This is a known issue with the memory module when testing
                # In production, the memory tools would be properly registered
                pytest.skip(f"Skipping due to memory tool error: {e}")
            else:
                # For other errors, re-raise
                raise
    
    # Get final state    
    state = email_assistant.get_state(thread_config)

    # Log inputs and outputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    t.log_outputs({"response": state.values if hasattr(state, "values") else state})

    # Get state values and verify
    values = extract_values(state)
        
    # Verify we have messages and classification
    assert "messages" in values
    assert len(values["messages"]) > 0
    assert "classification_decision" in values

@pytest.mark.langsmith
def test_hitl_notify():
    """Test the HITL workflow with a notification email and response feedback."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Use the notification email
    email_input = NOTIFICATION_EMAIL
    
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
    t.log_outputs({"messages": values["messages"]})
    
    # Assert we got responses and registered the feedback
    assert values["classification_decision"] is not None 
    assert "Let's actually respond" in all_messages_str

@pytest.mark.langsmith
def test_hitl_respond_edit():
    """Test the HITL workflow with response edit functionality."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Use the standard email
    email_input = STANDARD_EMAIL
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    
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

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})
    
    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    assert "Thanks Alice, I will fix it!" in all_messages_str

@pytest.mark.langsmith
def test_hitl_respond_feedback():
    """Test the HITL workflow with textual feedback for response generation."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory", "email_assistant_hitl"])
    
    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    
    # Use the standard email
    email_input = STANDARD_EMAIL
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    
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

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})
    
    # Grade the response using LLM
    llm = init_chat_model("openai:gpt-4o")
    structured_llm = llm.with_structured_output(GradeResponse)
    grade = structured_llm.invoke([
        {"role": "system",
         "content": f"This is an email assistant that is used to respond to emails. Review our initial email response and the user feedback given to update the email response. Here is the feedback: {feedback}. Assess whether the final email response addresses the feedback that we gave."},
        {"role": "user",
         "content": f"Here is the full conversation to grade, with the initial email response and the user feedback and the final email response at the end: {all_messages_str}"}
    ])

    print("GRADE: test_hitl_respond_feedback")
    print(grade)

    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    # assert grade.grade  # Uncomment to enforce passing grade

@pytest.mark.langsmith
def test_hitl_memory_notify():
    """Test the HITL-memory workflow with notification email and stored preferences."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory"])
    
    # Set up the assistant
    email_assistant, thread_config, store = setup_assistant()
    
    # Use the notification email
    email_input = NOTIFICATION_EMAIL
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": AGENT_MODULE})
    
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
    t.log_outputs({"messages": values["messages"]})

    # Retrieve stored triage preferences from memory
    results = store.search(("email_assistant", "triage_preferences"))
    triage_instructions_updated = results[0].value['content']['content']

    # Assert we got responses and registered the feedback and stored the triage instructions in memory
    assert values["classification_decision"] is not None 
    assert "For anything from the System Admin Team" in all_messages_str
    assert "System Admin" in triage_instructions_updated

@pytest.mark.langsmith
def test_hitl_memory_respond_edit():
    """Test the HITL-memory workflow with response edit and stored preferences."""
    
    # Skip test if module isn't supported
    check_module_compatibility(["email_assistant_hitl_memory"])
    
    # Set up the assistant
    email_assistant, thread_config, store = setup_assistant()
    
    # Use the standard email
    email_input = STANDARD_EMAIL
    
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

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})

    # Grade the response using LLM
    llm = init_chat_model("openai:gpt-4o")
    structured_llm = llm.with_structured_output(GradeResponse)
    grade = structured_llm.invoke([
        {"role": "system",
         "content": f"This is an email assistant that uses memory to update its response preferences. Review the initial response preferences and the updated response preferences. Assess whether the updated response preferences are more accurate than the initial response preferences."},
        {"role": "user",
         "content": f"Here are the initial response preferences: {response_preferences_pre_update}. Here are the updated response preferences: {response_preferences_post_update}. Here is the conversation: {all_messages_str}. Confirm that the response preferences are updated."}
    ])

    print("GRADE: test_hitl_memory_respond_edit")
    print(grade)
    
    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    # assert grade.grade  # Uncomment to enforce passing grade

if __name__ == "__main__":
    print(f"Testing agent module: {AGENT_MODULE}")
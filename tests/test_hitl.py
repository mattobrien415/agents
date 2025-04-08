import pytest
import uuid
import importlib
import sys
import os
from langsmith import testing as t
from pydantic import BaseModel, Field
from typing import Literal

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langchain.chat_models import init_chat_model

class GradeResponse(BaseModel):
    """Score the response."""
    grade: bool = Field(description="Does the response meet the criteria?")
    justification: str = Field(description="The justification for the grade.")

@pytest.mark.parametrize("module_name", ["email_assistant_hitl", "email_assistant_hitl_memory"])
@pytest.mark.langsmith
def test_hitl_notify(module_name):
    """Test that the HITL-enabled email assistant notification workflow runs successfully."""
    
    # Dynamically import module
    module = importlib.import_module(f"src.email_assistant.{module_name}")
    overall_workflow = module.overall_workflow
    
    # Test email input (notify scenario)
    email_input = {
        "author": "System Admin <sysadmin@company.com>",
        "to": "Development Team <dev@company.com>",
        "subject": "Scheduled maintenance - database downtime",
        "email_thread": """Hi team,

This is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.

Please plan your work accordingly and ensure no critical deployments are scheduled during this window.

Thanks,
System Admin Team"""
    }
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": module_name})
    
    # Compile the graph with store for memory version
    checkpointer = MemorySaver()
    if module_name == "email_assistant_hitl_memory":
        store = InMemoryStore()
        graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    else:
        graph = overall_workflow.compile(checkpointer=checkpointer)
    
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph initially
    messages = []
    for chunk in graph.stream({"email_input": email_input}, config=thread_config):
        messages.append(chunk)
    
    # Provide feedback and resume the graph
    resume_command = Command(resume=[{
        "type": "response", 
        "args": "Let's actually respond to this email with just a polite 'thank you!'"
    }])
    
    for chunk in graph.stream(resume_command, config=thread_config):
        messages.append(chunk)
    
    state = graph.get_state(thread_config)
    
    # Access state correctly based on type
    if hasattr(state, "values"):
        values = state.values
    else:
        values = state
    
    # Generate message output string
    all_messages_str = " === ".join(m.content for m in values['messages'])

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})
    
    # Assert we got responses and registered the feedback
    assert values["classification_decision"] is not None 
    assert "Let's actually respond" in all_messages_str

@pytest.mark.parametrize("module_name", ["email_assistant_hitl", "email_assistant_hitl_memory"])
@pytest.mark.langsmith
def test_hitl_respond_edit(module_name):
    """Test that the HITL-enabled email assistant response workflow runs successfully."""
    
    # Dynamically import module
    module = importlib.import_module(f"src.email_assistant.{module_name}")
    overall_workflow = module.overall_workflow
    
    # Test email input (respond scenario)
    email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",}
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": module_name})
    
    # Compile the graph with store for memory version
    checkpointer = MemorySaver()
    if module_name == "email_assistant_hitl_memory":
        store = InMemoryStore()
        graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    else:
        graph = overall_workflow.compile(checkpointer=checkpointer)
    
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph initially
    messages = []
    for chunk in graph.stream({"email_input": email_input}, config=thread_config):
        messages.append(chunk)
    
    # Edit the email and resume the graph
    resume_command = Command(resume=[{"type": "response",  
                                            "args": {"args": {"to": "Alice Smith <alice.smith@company.com>",
                                                           "subject": "RE: Quick question about API documentation",
                                                           "content": "Thanks Alice, I will fix it!"}}}])
    
    for chunk in graph.stream(resume_command, config=thread_config):
        messages.append(chunk)
    
    state = graph.get_state(thread_config)
    
    # Access state correctly based on type
    if hasattr(state, "values"):
        values = state.values
    else:
        values = state
    
    # Generate message output string
    all_messages_str = " === ".join(m.content for m in values['messages'])

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})
    
    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    assert "Thanks Alice, I will fix it!" in all_messages_str

@pytest.mark.parametrize("module_name", ["email_assistant_hitl", "email_assistant_hitl_memory"])
@pytest.mark.langsmith
def test_hitl_respond_feedback(module_name):
    """Test that the HITL-enabled email assistant response workflow runs successfully."""
    
    # Dynamically import module
    module = importlib.import_module(f"src.email_assistant.{module_name}")
    overall_workflow = module.overall_workflow
    
    # Test email input (respond scenario)
    email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",}
    
    # Log inputs
    t.log_inputs({"email_input": email_input, "module": module_name})
    
    # Compile the graph with store for memory version
    checkpointer = MemorySaver()
    if module_name == "email_assistant_hitl_memory":
        store = InMemoryStore()
        graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    else:
        graph = overall_workflow.compile(checkpointer=checkpointer)
    
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph initially
    messages = []
    for chunk in graph.stream({"email_input": email_input}, config=thread_config):
        messages.append(chunk)
    
    # Edit the email and resume the graph
    feedback = "Let's just say that we will fix it!'"
    resume_command = Command(resume=[{
        "type": "response", 
        "args": feedback
    }])
    
    for chunk in graph.stream(resume_command, config=thread_config):
        messages.append(chunk)
    
    state = graph.get_state(thread_config)
    
    # Access state correctly based on type
    if hasattr(state, "values"):
        values = state.values
    else:
        values = state
    
    # Generate message output string
    all_messages_str = " === ".join(m.content for m in values['messages'])

    # Log feedback response
    t.log_outputs({"messages": values["messages"]})
    
    # Grade the response
    llm = init_chat_model("openai:gpt-4o")
    structured_llm = llm.with_structured_output(GradeResponse)
    grade = structured_llm.invoke([{"role": "system",
                                    "content": f"This is an email assistant that is used to respond to emails. Review our initial email response and the user feedback given to update the email response. Here is the feedback: {feedback}. Assess whether the final email response addresses the feedback that we gave."},
                                    {"role": "user",
                                    "content": f"Here is the full conversation to grade, with the initial email response and the user feedback and the final email response at the end: {all_messages_str}"}])

    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    assert grade.grade
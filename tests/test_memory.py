import pytest
import uuid
import sys
import os
from langsmith import testing as t
from pydantic import BaseModel, Field
from typing import Literal

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use email_assistant_hitl_memory for memory tests
from src.email_assistant.email_assistant_hitl_memory import overall_workflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langchain.chat_models import init_chat_model

class GradeResponse(BaseModel):
    """Score the response."""
    grade: bool = Field(description="Does the response meet the criteria?")
    justification: str = Field(description="The justification for the grade.")


@pytest.mark.langsmith
def test_hitl_memory_notify():
    """Test that the HITL-enabled email assistant notification workflow runs successfully."""
    
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
    t.log_inputs({"email_input": email_input})
    
    # Compile the graph
    checkpointer = MemorySaver()
    store = InMemoryStore()
    graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph initially
    messages = []
    for chunk in graph.stream({"email_input": email_input}, config=thread_config):
        messages.append(chunk)
    
    # Provide feedback and resume the graph
    resume_command = Command(resume=[{
        "type": "response", 
        "args": "For anything from the System Admin Team, respond with just a polite 'thank you!'"
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

    # Results
    results = store.search(("email_assistant", "triage_preferences"))
    print("RESULTS - triage_preferences - FROM MEMORY")
    print(results)
    triage_instructions_updated = results[0].value['content']['content']

    # Assert we got responses and registered the feedback and stored the triage instructions in memory
    assert values["classification_decision"] is not None 
    assert "For anything from the System Admin Team" in all_messages_str
    assert "System Admin" in triage_instructions_updated

@pytest.mark.langsmith
def test_hitl_memory_respond_edit():
    """Test that the HITL-enabled email assistant response-edit workflow runs successfully."""
    
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
    t.log_inputs({"email_input": email_input})
    
    # Compile the graph
    checkpointer = MemorySaver()
    store = InMemoryStore()
    graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph initially
    messages = []
    for chunk in graph.stream({"email_input": email_input}, config=thread_config):
        messages.append(chunk)
    
    results = store.search(("email_assistant", "response_preferences"))
    response_preferences_pre_update = results[0].value['content']['content']

    # Edit the email and resume the graph
    resume_command = Command(resume=[{"type": "edit",  
                                         "args": {"args": {"to": "Alice Smith <alice.smith@company.com>",
                                                           "subject": "RE: Quick question about API documentation",
                                                           "content": "Thanks Alice, I will fix it!"}}}])
    
    for chunk in graph.stream(resume_command, config=thread_config):
        messages.append(chunk)
    
    results = store.search(("email_assistant", "response_preferences"))
    response_preferences_post_update = results[0].value['content']['content']

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
                                  "content": f"This is an email assistant that uses memory to update its response preferences. Review the initial response preferences and the updated response preferences. Assess whether the updated response preferences are more accurate than the initial response preferences."},
                                  {"role": "user",
                                  "content": f"Here are the initial response preferences: {response_preferences_pre_update}. Here are the updated response preferences: {response_preferences_post_update}. Here is the conversation: {all_messages_str}. Confirm that the response preferences are updated."}])

    
    # Assert we got responses and registered the edit
    assert values["classification_decision"] is not None 
    assert grade.grade
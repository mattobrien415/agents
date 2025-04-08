import pytest
import importlib
import uuid
from langsmith import testing as t

from langgraph.types import Command
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

@pytest.mark.parametrize("module_name", [
    "email_assistant_react",
    "email_assistant", 
    "email_assistant_hitl",
    "email_assistant_hitl_memory"
])

@pytest.mark.langsmith
def test_run(module_name):

    """Test that an email assistant implementation runs successfully."""    
    
    # Dynamically import the email assistant module
    module = importlib.import_module(f"src.email_assistant.{module_name}")
    overall_workflow = module.overall_workflow
    
    # Test email input that we will respond to
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
Alice""",
    }
    
    # Set up checkpointer and store
    checkpointer = MemorySaver()
    store = InMemoryStore()
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Compile the graph
    if module_name == "email_assistant_hitl_memory":
        # Memory implementation needs a store and a checkpointer
        graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
    else:
        # Just use a checkpointer
        graph = overall_workflow.compile(checkpointer=checkpointer)
    
    # Run the agent based on the module type
    if module_name == "email_assistant_react":
        # React agent takes messages
        messages = [{"role": "user", "content": str(email_input)}]
        result = graph.invoke({"messages": messages}, config=thread_config)

    elif module_name == "email_assistant":
        # Workflow agent takes email_input directly
        result = graph.invoke({"email_input": email_input}, config=thread_config)

    else:
        # Other agents take email_input directly but will use interrupt 
        result = {}
        for chunk in graph.stream({"email_input": email_input}, config=thread_config):
            result.update(chunk)
        # Provide feedback and resume the graph
        resume_command = Command(resume=[{
            "type": "accept", 
            "args": ""
        }])
        # Complete the graph
        for chunk in graph.stream(resume_command, config=thread_config):
            result.update(chunk)
    
    # Get final state    
    state = graph.get_state(thread_config)

    # Log inputs and outputs
    t.log_inputs({"email_input": email_input, "module": module_name})
    t.log_outputs({"response": state.values if hasattr(state, "values") else state})

    # Assert that we got a response
    assert hasattr(state, "values") or isinstance(state, dict)
    
    # Access state correctly depending on type
    if hasattr(state, "values"):
        values = state.values
    else:
        values = state
        
    # Verify we have messages and classification
    assert "messages" in values
    assert len(values["messages"]) > 0
    assert "classification_decision" in values
import pytest
from langsmith import testing as t
from src.email_assistant.email_assistant import email_assistant


@pytest.mark.langsmith
def test_email_assistant_workflow_runs():
    """Test that the workflow-based email assistant runs successfully."""
    
    # Test email input
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
    
    # Run the agent
    response = email_assistant.invoke({"email_input": email_input})
    
    # Log inputs and outputs
    t.log_inputs({"email_input": email_input})
    t.log_outputs({"response": response})
    
    # Assert that we got a response
    assert response is not None
    assert "messages" in response
    assert len(response["messages"]) > 0
    assert response["classification_decision"] is not None 
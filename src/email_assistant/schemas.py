from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.graph import MessagesState

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict

class State(MessagesState):
    # This state class has the messages key build in
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]

# User profile information
profile = {
    "name": "Lance",
    "full_name": "Lance Johnson",
    "user_profile_background": """
Lance is a senior software engineer at LangChain specializing in AI/ML systems.
He manages a team of 5 engineers working on the LangGraph project.
His time is valuable and he prefers concise communications.
    """,
    "email": "lance@langchain.com",
    "position": "Senior Software Engineer",
    "team": "LangGraph Team"
}
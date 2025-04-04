from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from src.email_assistant.prompts import agent_system_prompt_react
from src.email_assistant.schemas import State

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Initialize the LLM
llm = init_chat_model("openai:o3-mini")

# Agent tools 
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

@tool
def triage_email(category: Literal["ignore", "notify", "respond"]) -> str:
    """Triage an email into one of three categories: ignore, notify, respond."""
    return f"Classification Decision: {category}"

# Agent tools default
tools = [write_email, schedule_meeting, check_calendar_availability, triage_email]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# LLM call node
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    instructions = """
When handling emails, follow these steps:
1. Carefully analyze the email content and purpose
2. Triage emails using the triage_email tool to categorize as ignore, notify, or respond
3. For meeting requests, use check_calendar_availability to find open time slots
4. Schedule meetings with the schedule_meeting tool when appropriate
5. Draft response emails using the write_email tool
6. Always use professional and concise language
"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_react.format(
                        instructions=instructions
                    )}
                ]
                + state["messages"]
            )
        ]
    }

# Tool handling node
def tool_handler(state: State):
    """Process tool calls and execute them"""
    
    # Store messages
    result = []
    # Track if we need to update classification_decision
    classification_decision = None
    
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Execute the tool
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        # Log the classification decision if triage_email was called
        if tool_call["name"] == "triage_email":
            classification_decision = tool_call["args"]["category"]
            
        # Add the tool response
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    
    # Return updated state with both messages and classification_decision if available
    if classification_decision:
        return {
            "messages": result,
            "classification_decision": classification_decision
        }
    else:
        return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["tool_handler", END]:
    """Route to tool handler if tool call made, otherwise end"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_handler"
    return END

# Build workflow
agent_builder = StateGraph(State,input=MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_handler", tool_handler)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_handler": "tool_handler",
        END: END,
    },
)
agent_builder.add_edge("tool_handler", "llm_call")

# Compile the agent
email_assistant = agent_builder.compile()
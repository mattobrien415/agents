from typing import Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.types import interrupt 
  
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_memory, prompt_instructions
from email_assistant.schemas import State, RouterSchema, profile, StateInput
from email_assistant.utils import parse_email, format_for_display, format_email_markdown
from email_assistant.configuration import Configuration

from langmem import create_manage_memory_tool, create_search_memory_tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

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
class Question(BaseModel):
      """Question to ask user."""
      content: str

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Router LLM 
llm_router = llm.with_structured_output(RouterSchema) 

# Response agent prompt and tools  
memory_search_tool = create_search_memory_tool(namespace=("email_assistant", "response_preferences"))
tools = [write_email, schedule_meeting, check_calendar_availability, Question, memory_search_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# TODO: Implement reflection and save to memory 
def save_to_memory(state, tool_call, action_type, edited_content=None):
    """Save interaction to memory for learning"""
    # Implementation depends on your memory system
    pass

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Note: We don't need to store email details separately
    # email_input is already in the state and will be available to the response agent
    
    # Create email markdown for Agent Inbox using the utility function
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision  
    messages = [{"role": "user",
                "content": f"Classification Decision: {result.classification}"
                }]

    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Add the email to the messages
        messages.append({"role": "user",
                            "content": f"Respond to the email {state['email_input']}"
                            })
        # Update the state
        update = {
            "messages": messages,
            "classification_decision": result.classification,
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        # Next node
        goto = END
        # Update the state
        update = {
            "messages": messages,
            "classification_decision": result.classification,
        }
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")        
        # Create interrupt for Agent Inbox
        request = {
            "action_request": {
                "action": f"Email Assistant: {result.classification}",
                "args": {}
            },
            "config": {
                "allow_ignore": False,  
                "allow_respond": True, # Allow user feedback if decision is not correct 
                "allow_edit": False, 
                "allow_accept": True, # Allow user to accept decision 
            },
            # Email to show 
            "description": email_markdown,
        }
        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]
        # Accept the decision to respond  
        if response["type"] == "accept":
            goto = END 
        elif response["type"] == "response":
            # Add feedback to messages 
            user_input = response["args"]
            messages.append({"role": "user",
                            "content": f"User feedback: {user_input}"
                            })
            # TODO: Go to reflection node when we add memory
            goto = END # save_to_memory

        # Update the state 
        update = {
            "messages": messages,
            "classification_decision": result.classification,
        }
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [ # TODO: prompt_instructions should come from memory 
                    SystemMessage(
                        content=agent_system_prompt_memory.format(instructions=prompt_instructions["agent_instructions"], **profile)
                    )
                ]
                + state["messages"]
            )
        ]
    }

def interrupt_handler(state: State):
    """Creates an interrupt for human review of tool calls"""
    
    result = []

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        
        # List of tools that require human intervention
        # TODO: Is this best practice? Ugly, but needed to bypass 
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        
        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:
            # Execute search_memory and other tools without interruption
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            continue
            
        # Get original email from email_input in state
        original_email_markdown = ""
        if "email_input" in state:
            email_input = state["email_input"]
            author, to, subject, email_thread = parse_email(email_input)
            original_email_markdown = format_email_markdown(subject, author, to, email_thread)
        
        # Format tool call for display and prepend the original email
        tool_display = format_for_display(state, tool_call)
        description = original_email_markdown + tool_display

        # Configure what actions are allowed in Agent Inbox
        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }

        # Create the interrupt request
        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": config,
            "description": description,
        }

        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]

        # Handle the response
        if response["type"] == "accept":
            # Execute the tool with original args
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            # TODO: Add accepted tool call to memory  
            # if config.get("memory_enabled", False):
            #    save_to_memory(state, tool_call, "accepted")

        elif response["type"] == "edit":
            # Execute with edited args
            tool = tools_by_name[tool_call["name"]]
            edited_args = response["args"]["args"]
            observation = tool.invoke(edited_args)
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            # TODO: Save edits to memory
            # if config.get("memory_enabled", False):
            #    save_to_memory(state, tool_call, "edited", edited_args)

        elif response["type"] == "ignore":
            # Don't execute the tool
            result.append(ToolMessage(content="Tool execution cancelled by user", tool_call_id=tool_call["id"]))

        elif response["type"] == "response":
            # User provided feedback
            result.append(ToolMessage(content=f"Feedback: {response['args']}", tool_call_id=tool_call["id"]))
            # TODO: Save feedback to memory
            # if config.get("memory_enabled", False):
            #    save_to_memory(state, tool_call, "feedback", response["args"])

    return {"messages": result}

# Conditional edge functions
def should_continue(state: State) -> Literal["interrupt_handler", END]:
    """Route to interrupt handler if tool call made"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "interrupt_handler"
    return END

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)
agent_builder.add_edge("interrupt_handler", "llm_call")

# Compile the agent
response_agent = agent_builder.compile()

# We no longer need a separate persist function
# email_input is automatically passed from the outer graph to the inner graph

# Build overall workflow
email_assistant = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    .compile()
)
from typing import TypedDict, Literal, Optional, Union
import yaml
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
 
from langgraph.types import interrupt 
  
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, prompt_instructions
from email_assistant.schemas import State, RouterSchema, profile
from email_assistant.utils import parse_email

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# We'll use structured output to generate classification results
llm_router = llm.with_structured_output(RouterSchema) 

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

def create_prompt(state):
    return [
        {"role": "system", "content": agent_system_prompt.format(instructions=prompt_instructions["agent_instructions"], **profile)}
    ] + state['messages']

# Baseline agent prompt
prompt = agent_system_prompt
tools = [write_email, schedule_meeting, check_calendar_availability]

# Create response agent
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=prompt,
)

def triage_router_hitl(state: State) -> Command[Literal["response_agent", "__end__"]]:
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

    # Create email markdown for Agent Inbox
    email_markdown = f""" 
    Subject: {subject}
    Author: {author}
    To: {to}
    From: {author}
    Email Thread:
    {email_thread}
    ==== ==== ====
    """

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        decision = "📧 Classification: RESPOND - This email requires a response"
        print(decision)
        email_markdown = f"# Review Decision: {result.classification}" + "\n" + email_markdown + "\n" + decision

        # Create messages for Agent Inbox
        messages = [{"role": "user",
                     "content": f"Classification Decision: {result.classification}"
                     }]

        # Create interrupt for Agent Inbox
        request = {
            "action_request": {
                "action": f"Review Triage: {result.classification}",
                "args": {}
            },
            "config": {
                "allow_ignore": False, # Don't display ignore button 
                "allow_respond": True, # Allow user feedback if decision is not correct 
                "allow_edit": False, # Nothing to edit at the triage step 
                "allow_accept": True, # Allow user to accept decision 
            },
            # Email to show 
            "description": email_markdown,
        }

        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]
        print("***Agent Inbox response:***", response)

        # Accept the decision to respond  
        if response["type"] == "accept":
            # Go to the response agent
            goto = "response_agent"
            # Add the email to the messages
            messages.append({"role": "user",
                             "content": f"Respond to the email {state['email_input']}"
                             })
        # Ignore the email 
        elif response["type"] == "ignore":
            # TODO: Add decision to memory that user preference differs from decision made by assistant 
            goto = END
            # Add the email to the messages
            messages.append({"role": "user",
                             "content": "User feedback: Ignore email"
                             })
        elif response["type"] == "response":
            # TODO: Add user_input to memory that user preference differs from decision made by assistant
            user_input = response["args"]
            messages.append({"role": "user",
                             "content": f"User feedback: {user_input}"
                             })
            goto = END

        # Update the state 
        update = {
            "messages": messages,
            "classification_decision": result.classification,
        }

    elif result.classification == "ignore":
        decision = "🚫 Classification: IGNORE - This email can be safely ignored"
        print(decision)
        email_markdown = f"# Review Decision: {result.classification}" + "\n" + email_markdown + "\n" + decision

        # Create messages for Agent Inbox
        messages = [{"role": "user",
                     "content": f"Classification Decision: {result.classification}"
                     }]

        # Create interrupt for Agent Inbox
        request = {
            "action_request": {
                "action": f"Review Triage: {result.classification}",
                "args": {}
            },
            "config": {
                "allow_ignore": False, # TODO: Check UI? 
                "allow_respond": True, # Allow user feedback if decision is not correct 
                "allow_edit": False, 
                "allow_accept": True, # Allow user to accept decision 
            },
            # Email to show 
            "description": email_markdown,
        }

        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]
        print("***Agent Inbox response:***", response)

        # Accept the decision to ignore  
        if response["type"] == "accept":
            goto = END
        # Respond to the email instead 
        elif response["type"] == "response":
            # TODO: Add memory update to log that user preference differs from decision made by assistant w/ feedback 
            user_input = response["args"]
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
        
    elif result.classification == "notify":
        decision = "🔔 Classification: NOTIFY - This email contains important information"
        print(decision)
        email_markdown = f"# Review Decision: {result.classification}" + "\n" + email_markdown + "\n" + decision
        
        # Create messages for Agent Inbox
        messages = [{"role": "user",
                     "content": f"Classification Decision: {result.classification}"
                     }]
        
        # Create interrupt for Agent Inbox
        request = {
            "action_request": {
                "action": f"Review Triage: {result.classification}",
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
        print("***Agent Inbox response:***", response)
        
        # Accept the decision to respond  
        if response["type"] == "accept":
            goto = "response_agent"
            # Add the email to the messages
            messages.append({"role": "user",
                            "content": f"Respond to the email {state['email_input']}"
                            })
        # Ignore the email 
        elif response["type"] == "ignore":
            # TODO: Add memory update to log that user preference differs from decision made by assistant 
            goto = END
            # Add the user feedback to messages
            messages.append({"role": "user",
                            "content": "User feedback: Ignore email"
                            })
        elif response["type"] == "response":
            # TODO: Add memory update to log that user preference differs from decision made by assistant w/ feedback 
            user_input = response["args"]
            messages.append({"role": "user",
                            "content": f"User feedback: {user_input}"
                            })
            goto = END

        # Update the state 
        update = {
            "messages": messages,
            "classification_decision": result.classification,
        }

    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

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

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [{
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}",
                },
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ],
            "classification_decision": result.classification,
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update =  {
            "messages": [
                {
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}"
                }
            ],
            "classification_decision": result.classification,
        }
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Classification Decision: {result.classification}"
                }
            ],
            "classification_decision": result.classification,
        }
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# Build workflow
email_assistant = (
    StateGraph(State)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
    .compile()
)

# Build workflow w/ HITL 
email_assistant_hitl = (
    StateGraph(State)
    .add_node(triage_router_hitl)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router_hitl")
    .compile()
)
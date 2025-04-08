from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
   
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_email_markdown

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o")

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

# Baseline agent prompt
tools = [write_email, schedule_meeting, check_calendar_availability]

# Create response agent
agent = create_react_agent(
    llm,
    tools=tools,
    prompt= agent_system_prompt.format(background=default_background,
                                       response_preferences=default_response_preferences, 
                                       cal_preferences=default_cal_preferences),
)

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
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
                    "content": f"Respond to the email: \n\n{format_email_markdown(subject, author, to, email_thread)}",
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
overall_workflow = (
    StateGraph(State,input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)

email_assistant = overall_workflow.compile()
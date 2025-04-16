from typing import Literal
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from langmem import create_memory_store_manager

from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl_memory, default_triage_instructions, default_background, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_for_display, format_email_markdown

# Agent tools 
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

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

@tool
class Done(BaseModel):
      """E-mail has been sent."""
      done: bool
    
# All tools available to the agent
tools = [
    write_email, 
    schedule_meeting, 
    check_calendar_availability, 
    Question, 
    Done
]

tools_by_name = {tool.name: tool for tool in tools}

# Initialize the LLM for use with router / structured output
llm = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = init_chat_model("openai:gpt-4o", tool_choice="required", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# TODO For string profiles consider just using the memory store directly to store the strings
# Could use prompt optimizer to update them: 
# https://langchain-ai.github.io/langmem/reference/prompt_optimization/#langmem.create_prompt_optimizer
triage_feedback_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "triage_preferences"),
    instructions="""Extract user email triage preferences into a set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False, # Do not delete profile from memory
    default=default_triage_instructions
)

response_preferences_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "response_preferences"),
    instructions="""Extract a profile that contains a user's email response preferences. 
    If you are given a set of rules, do not remove any rules and simply include them in the resulting profile.
    If you are given feedback on an email response, update the profile to reflect the new preferences.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False, # Do not delete profile from memory
    default=default_response_preferences
)

cal_preferences_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "cal_preferences"),
    instructions="""Extract user email calendar preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False, # Do not delete profile from memory
    default=default_cal_preferences
)   

background_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "background"),
    instructions="""Extract user email background information about the user, their key connections, and other relevant information.
    Format this as a collection of short memories that can be easily recalled.""",
    enable_inserts=True, # Update background in-place,
    enable_deletes=True, # Since this is a collection, we can delete items (if they are no longer relevant)
    default=default_background
)

# Nodes 
def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    
    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Search for existing triage_preferences memory
    result = triage_feedback_memory_manager.search()
    triage_instructions=result[0].value.content
    
    # Search for existing background memory
    results = background_memory_manager.search(query=email_markdown)

    # Handle collection of memory objects
    memories = []
    for result in results:
        memories.append(result.value.content)
    background = "\n".join(memories)
        
    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=background,
        triage_instructions=triage_instructions,
    )

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Update the state
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
        # TODO: We may move this  
        # Remember facts about this email because we've decided to respond to it, so we deem it to be important
        background_memory_manager.invoke({"messages": [{"role": "user", "content": f"The user will reply to this email, so remember relevant facts about it: {email_markdown}"}]})

    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "classification_decision": classification,
        }

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information") 

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")
    
    return Command(goto=goto, update=update)

def triage_interrupt_handler(state: State, store: BaseStore) -> Command[Literal["response_agent", "__end__"]]:
    """Handles interrupts from the triage step"""
    
    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Create messages
    messages = [{"role": "user",
                "content": f"Email to notify user about: {email_markdown}"
                }]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,  
            "allow_respond": True,
            "allow_edit": False, 
            "allow_accept": False,  
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Send to Agent Inbox and wait for response
    response = interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email   
    if response["type"] == "response":
        # Add feedback to messages 
        user_input = response["args"]
        messages.append({"role": "user",
                        "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"
                        })
        # Update memory with feedback using the memory manager
        triage_feedback_memory_manager.invoke({"messages": messages})

        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        # Make note of the user's decision to ignore the email
        messages.append({"role": "user",
                        "content": f"The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this."
                        })
        # Update memory with feedback using the memory manager
        triage_feedback_memory_manager.invoke({"messages": messages})
        goto = END

    # Catch all other responses
    else:
        raise ValueError(f"Invalid response: {response}")

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)

def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not"""

    # Search for existing cal_preferences memory
    result = cal_preferences_memory_manager.search()
    cal_preferences=result[0].value.content
    
    # Search for existing response_preferences memory
    result = response_preferences_memory_manager.search()
    response_preferences=result[0].value.content

    # Create email markdown and search for existing background memory
    author, to, subject, email_thread = parse_email(state["email_input"])
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    results = background_memory_manager.search(query=email_markdown)

    # Handle collection of memory objects
    memories = []
    for result in results:
        memories.append(result.value.content)
    background = "\n".join(memories)

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl_memory.format(background=background,
                                                                                         response_preferences=response_preferences, 
                                                                                         cal_preferences=cal_preferences)}
                ]
                + state["messages"]
            )
        ]
    }
    
def interrupt_handler(state: State, store: BaseStore):
    """Creates an interrupt for human review of tool calls"""
    
    # Store messages
    result = []

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        
        # Allowed tools for HITL
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        
        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:

            # Execute search_memory and other tools without interruption
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue
            
        # Get original email from email_input in state
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
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

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

        # Handle the responses 
        if response["type"] == "accept":

            # Execute the tool with original args
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
                        
        elif response["type"] == "edit":

            # Tool selection 
            tool = tools_by_name[tool_call["name"]]
            
            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Save feedback in memory and update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # We update the memory in the namespace with the messages from the state
                response_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is a better way to respond to emails: {edited_args}"}]
                })
                
                # Update the AI message's tool call with edited content (reference to the message in the state)
                ai_message = state["messages"][-1]
                current_id = tool_call["id"]
                
                # Replace the original tool call with the edited one (any changes made to this reference affect the original object in the state)
                ai_message.tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                    {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
                ]
                
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
            
            # Save feedback in memory and update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting":
                # Add context about calendar preferences
                # todo make sure this works
                cal_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here are preferred calendar settings: {edited_args}"}]
                })
                
                # Update the AI message's tool call with edited content
                ai_message = state["messages"][-1]
                current_id = tool_call["id"]
                
                # Replace the original tool call with the edited one
                ai_message.tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                    {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
                ]
                
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            # Catch all other tool calls
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            # Update relevant domain-specific memory
            if tool_call["name"] == "write_email":
                # Add context about email response preferences
                triage_feedback_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User decided to ignore this email even though it was classified as respond, so update the triage preferences to capture this."}]
                })
            elif tool_call["name"] == "schedule_meeting":
                # Add context about calendar preferences
                cal_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User decided to ignore this email meeting request though it was classified as notify, so update the calendar preferences to capture this."}]
                })
            elif tool_call["name"] == "Question":
                # Add context about question
                background_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User ignored this question, so update the background information to capture this."}]
                })
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            result.append({"role": "tool", "content": f"Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            # Also update  memory
            if tool_call["name"] == "write_email":
                # Add context about email response preferences
                response_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is feedback on how to respond to the email: {user_feedback}"}]
                })
                background_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is user feedback on the email, so it to update the background information: {user_feedback}"}]
                })
            elif tool_call["name"] == "schedule_meeting":
                # Add context about calendar preferences
                cal_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is feedback on calendar scheduling: {user_feedback}"}]
                })
            elif tool_call["name"] == "Question":
                # Add context about question
                background_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User answered this question, so update the background information to capture this: {user_feedback}`"}]
                })
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

    return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["interrupt_handler", END]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"

# Build workflow
agent_builder = StateGraph(State)

# Add nodes - with store parameter
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

# Compile the agent - nodes will receive store parameter automatically
response_agent = agent_builder.compile()

# Build overall workflow with store and checkpointer
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
)

email_assistant = overall_workflow.compile()
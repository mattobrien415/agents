from typing import Literal
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from langmem import create_search_memory_tool, create_memory_store_manager

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
    
# Memory search tools for reading from memory Store 
response_preferences_tool = create_search_memory_tool(namespace=("email_assistant", "response_preferences"), name="response_preferences")
cal_preferences_tool = create_search_memory_tool(namespace=("email_assistant", "cal_preferences"), name="cal_preferences")
background_tool = create_search_memory_tool(namespace=("email_assistant", "background"), name="background")
triage_preferences_tool = create_search_memory_tool(namespace=("email_assistant", "triage_preferences"), name="triage_preferences")

# All tools available to the agent
tools = [
    write_email, 
    schedule_meeting, 
    check_calendar_availability, 
    Question, 
    response_preferences_tool, 
    cal_preferences_tool, 
    background_tool,
    triage_preferences_tool,
    Done
]

tools_by_name = {tool.name: tool for tool in tools}


# Initialize the LLM for use with router / structured output
llm = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 
# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = init_chat_model("openai:gpt-4o", tool_choice="required", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# Feedback memory managers for writing to memory Store 
triage_feedback_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "triage_preferences"),
    instructions="""Extract user email triage preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False # Do not delete profile from memory
)

response_preferences_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "response_preferences"),
    instructions="""Extract user email response preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False # Do not delete profile from memory
)

cal_preferences_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "cal_preferences"),
    instructions="""Extract user email calendar preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False # Do not delete profile from memory
)   

background_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "background"),
    instructions="""Extract user email background information about the user, their key connections, and other relevant information.
    Format this as a collection of short memories that can be easily recalled.""",
    enable_inserts=True, # Update background in-place,
    enable_deletes=True # Since this is a collection, we can delete items (if they are no longer relevant)
)

# Nodes 
def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """

    # TODO(discuss w/ Will):: FIND BETTER WAY TO HANDLE THIS; WE NEED TO INITIALIZE THE MEMORY IF IT DOESN'T EXIST
    results = store.search(("email_assistant", "triage_preferences"))
    if results:
        triage_instructions = results[0].value['content']['content']
    else:
        triage_instructions = default_triage_instructions
        triage_feedback_memory_manager.invoke({"messages": [{"role": "user", "content": triage_instructions}]})
        
    # Search for background information in Store
    results = store.search(("email_assistant", "background"))
    if results:
        # Handle background which could be a list of memory objects
        backgrounds = []
        for result in results:
            backgrounds.append(result.value['content']['content'])
        background_content = "\n".join(backgrounds)
    else:
        background_content = default_background
        background_memory_manager.invoke({"messages": [{"role": "user", "content": background_content}]})
        
    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=background_content,
        triage_instructions=triage_instructions,
    )

    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification
    messages = [{"role": "user",
                "content": f"Classification Decision: {classification}"
                }]

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Add the email to the messages
        messages.append({"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                            })
        # Update the state
        update = {
            "messages": messages,
            "classification_decision": classification,
        }
    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "messages": messages,
            "classification_decision": classification,
        }

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information") 

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "messages": messages,
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

    # Create messages to save to memory
    messages = [{"role": "user",
                "content": f"Classification Decision: {state['classification_decision']} for email: {email_markdown}"
                }]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,  
            "allow_respond": True, # Allow user feedback if decision is not correct 
            "allow_edit": False, 
            "allow_accept": False,  
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Send to Agent Inbox and wait for response
    response = interrupt([request])[0]

    # Accept the decision and end   
    if response["type"] == "accept":
        goto = END 

    # If user provides feedback, update memory  
    elif response["type"] == "response":
        # Add feedback to messages 
        user_input = response["args"]
        messages.append({"role": "user",
                        "content": f"Here is feedback on how the user would prefer the email to be classified: {user_input}"
                        })
        # Update memory with feedback using the memory manager
        triage_feedback_memory_manager.invoke({"messages": messages})
        goto = END

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)

def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not"""

    # TODO (discuss w/ Will): May be a nicer way to initialize the memory
    # Calendar preferences
    results = store.search(
        ("email_assistant", "cal_preferences")
    )
    if not results:
        cal_preferences = default_cal_preferences
        cal_preferences_memory_manager.invoke({"messages": [{"role": "user", "content": cal_preferences}]})

    # Response preferences
    results = store.search(
        ("email_assistant", "response_preferences")
    )
    if not results:
        response_preferences = default_response_preferences
        response_preferences_memory_manager.invoke({"messages": [{"role": "user", "content": response_preferences}]})

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl_memory}
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
        
        # TODO (discuss w/ Vadym): FIND BETTER WAY TO HANDLE THIS
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        
        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:
            # Execute search_memory and other tools without interruption
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
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

        # Handle the responses 
        if response["type"] == "accept":

            # Execute the tool with original args
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            
            # Remember facts from the conversation with background memory manager
            background_memory_manager.invoke({"messages": state["messages"] + result})
            
        elif response["type"] == "edit":

            # Tool selection 
            tool = tools_by_name[tool_call["name"]]
            
            # Get edited args from Agent Inbox
            # TODO: Talk to Brace. This is confusing. 
            edited_args = response["args"]["args"]

            # Save feedback in memory and update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # Let's save our response preferences in memory
                response_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is a better way to respond to emails: {edited_args}"}]
                })
                
                # TODO (discuss w/ Vadym): FIND BETTER WAY TO HANDLE THIS 
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

        elif response["type"] == "ignore":
            # Update relevant domain-specific memory
            if tool_call["name"] == "write_email":
                # Add context about email response preferences
                response_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User decided to ignore this email! Make note of this as an few shot example."}]
                })
            elif tool_call["name"] == "schedule_meeting":
                # Add context about calendar preferences
                cal_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"User decided to ignore this email! Make note of this as an few shot example."}]
                })

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            result.append({"role": "tool", "content": f"Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            # Also update relevant domain-specific memory
            if tool_call["name"] == "write_email":
                # Add context about email response preferences
                response_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is feedback on how to respond to emails: {user_feedback}"}]
                })
            elif tool_call["name"] == "schedule_meeting":
                # Add context about calendar preferences
                cal_preferences_memory_manager.invoke({
                    "messages": state["messages"] + [{"role": "user", "content": f"Here is feedback on calendar scheduling: {user_feedback}"}]
                })

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
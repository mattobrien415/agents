# Human-in-the-Loop Email Assistant

Building on our email assistant, we can enhance it with human-in-the-loop capabilities using LangGraph's interrupts. This approach integrates human oversight at critical decision points, providing improved control and ensuring the agent operates with appropriate human supervision.

![overview-img](img/overview_hitl.png)

## Human-in-the-Loop with LangGraph Interrupts

The HITL (Human-In-The-Loop) pattern is critical for applications where important decisions require human validation. LangGraph provides built-in support for this through its [interrupt mechanism](https://langchain-ai.github.io/langgraph/concepts/interrupts/), allowing us to pause execution and request human input when needed.

LangGraph's interrupt mechanism works in conjunction with [Agent Inbox](https://github.com/langchain-ai/agent-inbox), a user interface designed specifically for human-in-the-loop interactions. When our email assistant needs human input, it creates an interrupt request that appears in Agent Inbox, allowing humans to review, edit, or provide feedback on the agent's actions.

![hitl-img](img/hitl.png)

### Core Components of the HITL Email Assistant

The HITL email assistant builds on our previous implementation but adds interrupt points for human oversight:

```python
from typing import Literal
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_for_display, format_email_markdown
```

### Agent Tools with HITL Integration

Our tools remain similar to the baseline version, but we now add a `Question` tool that allows the agent to explicitly request human input:

```python
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
```

### Interrupt Handler for Triage

One key enhancement is adding human oversight at the triage stage. When an email is classified as "notify," we interrupt to get human confirmation:

```python
def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", "__end__"]]:
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

    # Agent Inbox responds with a list  
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

        goto = END

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)
```

The interrupt request structure follows Agent Inbox's expected format:
1. `action_request`: Defines what action the agent is trying to take
2. `config`: Specifies what interaction types are allowed (ignore, respond, edit, accept)
3. `description`: Provides context (in this case, the email content) shown in the Agent Inbox UI

This structure allows Agent Inbox to render appropriate UI controls and collect user feedback, which we'll be able to test with our local deployment later.

### Tool Execution Interrupt Handler

The main interrupt handler is responsible for pausing execution before executing certain tools (write_email, schedule_meeting, Question), allowing humans to review, edit, or reject these actions:

```python
def interrupt_handler(state: State):
    """Creates an interrupt for human review of tool calls"""
    
    # Store messages
    result = []

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        
        # Define which tools require human oversight
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
                        
        elif response["type"] == "edit":
            # Tool selection 
            tool = tools_by_name[tool_call["name"]]
            
            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Update the tool calls with edited content
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
            # Don't execute the tool
            result.append({"role": "tool", "content": "Tool execution cancelled by user", "tool_call_id": tool_call["id"]})
            
        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            result.append({"role": "tool", "content": f"Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            
    return {"messages": result}
```

### Complete Workflow with HITL

The complete workflow combines the triage router with interrupt handling and the response agent:

```python
# Build response agent
agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)
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
response_agent = agent_builder.compile()

# Build overall workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
)

email_assistant = overall_workflow.compile()
```

## Benefits of the HITL Approach

1. **Human Oversight**: Critical actions require human approval before execution
2. **Learning Opportunities**: Human feedback is captured and can be used for further training
3. **Error Prevention**: Humans can catch and correct mistakes before they happen
4. **Trust Building**: Users gain confidence in the system knowing they have final say on important actions
5. **Progressive Automation**: As the system proves reliable, oversight can be gradually reduced

This HITL implementation showcases how LangGraph's interrupt mechanism combined with Agent Inbox creates a powerful collaboration between human intelligence and AI capabilities, leading to more reliable and trustworthy agent systems.

## Testing with Local Deployment

As outlined in the README, we'll be able to test our HITL implementation by:

1. Running `langgraph dev` to start the LangGraph server locally
2. Connecting to Agent Inbox at https://dev.agentinbox.ai/
3. Adding a new inbox pointing to our local LangGraph server
4. Sending email inputs through LangGraph Studio to trigger interrupts
5. Responding to those interrupts in Agent Inbox

This setup will allow us to experience the full human-in-the-loop workflow, seeing how our interrupt requests render in Agent Inbox and how we can provide feedback that influences the assistant's behavior.
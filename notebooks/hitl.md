# Human-in-the-Loop

Our email assistant can be triage emails and use tools to respond to them. But do we actually trust it to manage our inbox? Few would trust an AI to manage their inbox without some human oversight at the start, which is why human-in-the-loop (HITL) is a critical pattern for agent systems.

![overview-img](img/overview_hitl.png)

## Human-in-the-Loop with LangGraph Interrupts

The HITL (Human-In-The-Loop) pattern is useful for applications where decisions require human validation. LangGraph provides built-in support for this through its [interrupt mechanism](https://langchain-ai.github.io/langgraph/concepts/interrupts/), allowing us to pause execution and request human input when needed. Let's add HITL to our email assistant after specific tools are called!

### Simple Interrupt Example

Let's assume we want a simple agent that can ask the user a question with a tool call and then use that information. The agent needs to stop and wait for the user to provide the information. This is where the `interrupt` function comes in. The `interrupt` function is the core of LangGraph's human-in-the-loop capability:

```
location = interrupt(ask.question)
```

When this line executes:
1. It raises a `GraphInterrupt` exception, which pauses the graph execution
2. It surfaces the value (the question) to the client
3. Execution stops at this point until resumed with a `Command`
4. When resumed, the function returns the value provided by the human

Here's a minimal example of how to implement a basic interrupt with an agent:

```python
from pydantic import BaseModel
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from IPython.display import Image, display

@tool
def search(query: str):
    """Call to surf the web."""
    return f"I looked up: {query}. Result: It's sunny in San Francisco."

# We can define a tool definition for `ask_human`
class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

tools = [search, AskHuman]
tool_node = ToolNode([search])

# Set up the model
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "action"

def call_model(state):
    messages = state["messages"]
    message = llm_with_tools.invoke(messages)
    return {"messages": [message]}

def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
    location = interrupt(ask.question)
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}

# Define a new graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
workflow.add_edge(START, "agent")
# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
# We now add a normal edge from `tools` to `agent`.
workflow.add_edge("action", "agent")
workflow.add_edge("ask_human", "agent")

# Set up memory
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
display(Image(app.get_graph().draw_mermaid_png()))
```

Now, we ask the user where they are and look up the weather there:

```python
config = {"configurable": {"thread_id": "1"}}
messages = [{"role": "user", "content": "Ask the user where they are, then look up the weather there"}]
for event in app.stream({"messages": messages}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

You can see that our graph got interrupted inside the ask_human node. 

It is now waiting for a location to be provided. 
```python
app.get_state(config).next
```

### Using Command to Resume Execution

After an interrupt, we need a way to continue execution. This is where the `Command` interface comes in. The `Command` object has several powerful capabilities:
- `resume`: Provides the value to return from the interrupt call
- `goto`: Specifies which node to route to next
- `update`: Modifies the state before continuing execution
- `graph`: Controls navigation between parent and child graphs

In this case, the `Command` object serves two crucial purposes:
1. It provides the value to be returned from the `interrupt` call
2. It controls the flow of execution in the graph

```python
# Resume execution with the value "san francisco"
for event in app.stream(Command(resume="san francisco"), config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

## Agent Inbox

While we can implement basic human-in-the-loop functionality using raw `interrupt` calls and Command responses, our email assistant benefits from a more structured interface for human interaction. This is especially important for an email assistant where we need *multiple types of human review*.

### HITL Interactions for Email Assistant

Our email assistant requires several types of human-in-the-loop interactions:

1. **Email Triage Review**:
   - When an email is classified as "NOTIFY," humans should verify this classification
   - Users should be able to accept the classification or provide feedback on how it should be classified

2. **Email Response Review**:
   - Before sending responses to important emails, humans should review the content
   - Users need options to edit the draft, accept it as-is, provide feedback, or reject it entirely

3. **Meeting Scheduling Review**:
   - When the assistant proposes scheduling a meeting, humans should verify the details
   - Users should be able to modify attendees, duration, date/time before accepting

4. **General Tool Execution Approval**:
   - Any significant action (sending emails, scheduling meetings) requires human approval
   - Some low-risk tools (like calendar availability checks) can run without interruption

### Agent Inbox: A Purpose-Built HITL Interface

To handle these complex interaction patterns, we use [Agent Inbox](https://github.com/langchain-ai/agent-inbox), a specialized interface for human-in-the-loop AI agents that integrates with LangGraph.

Agent Inbox provides:

1. **Structured Interaction Types**:
   - `accept`: Approve the agent's action and continue
   - `edit`: Modify the agent's proposed action before execution
   - `response`: Provide feedback or answers without editing
   - `ignore`: Reject the agent's action entirely

2. **Rich Content Display**:
   - Render (email) content in a readable format
   - Support markdown for structured information presentation

3. **Consistent User Experience**:
   - Notification system for pending reviews
   - Action buttons that match the allowed interaction types
   - Thread-based organization of agent activities

4. **Easy Integration with LangGraph**:
   - Simple connection to local or hosted LangGraph deployments
   - Compatible with LangGraph's interrupt mechanism
   - No complex frontend development required

### Integration with LangGraph's interrupt() Function

Agent Inbox integrates seamlessly with LangGraph's `interrupt()` function. The integration works like this:

1. **Request Structure**: We structure an interrupt request with specific fields:
   ```python
   request = {
       "action_request": {
           "action": "write_email",  # Name of the tool to call
           "args": {"to": "john@example.com", "subject": "Meeting", "content": "..."}  # Action parameters
       },
       "config": {
           "allow_ignore": True,   # Can dismiss the action
           "allow_respond": True,  # Can provide feedback
           "allow_edit": True,     # Can modify the action
           "allow_accept": True,   # Can approve the action
       },
       "description": "Email content to display..." # Context shown to the user
   }
   ```

2. **Passing to interrupt()**: We pass this request to the interrupt function:
   ```python
   response = interrupt([request])[0]  # Can batch multiple requests
   ```

3. **User Interaction**: Agent Inbox shows the request and collects the user's response

4. **Handling Responses**: When execution resumes, we receive a structured `response`:
   ```python
   if response["type"] == "accept":
       # Execute the tool with original args
   elif response["type"] == "edit":
       # Execute with edited args from response["args"]
   elif response["type"] == "ignore":
       # Skip execution
   elif response["type"] == "response":
       # Process feedback from response["args"]
   ```

This structured approach allows our email assistant to collect precise human input at critical decision points while maintaining a consistent user experience.

## Email Assistant with Human-in-the-Loop

Now that we understand both the interrupt mechanism and Agent Inbox, let's look at our complete email assistant implementation with human-in-the-loop capabilities. This implementation brings together all the concepts we've discussed:

1. It uses the `interrupt` function to pause execution at key decision points
2. It structures interrupt requests specifically for Agent Inbox
3. It processes different response types from human reviewers
4. It integrates these HITL capabilities into a full email processing workflow

The full implementation consists of:
- Tools for email management, meeting scheduling, and human interaction
- A triage router that categorizes incoming emails
- An interrupt handler for triage decisions (NOTIFY classification)
- A separate interrupt handler for tool execution (write_email, schedule_meeting, etc.)
- A complete workflow that connects these components

```python
%cd ..
%load_ext autoreload
%autoreload 2
```

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
    Done,
]

tools_by_name = {tool.name: tool for tool in tools}

# Initialize the LLM for use with router / structured output
llm = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 
# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = init_chat_model("openai:gpt-4o", tool_choice="required", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)
```

### Core Nodes of the Email Assistant

Our email assistant has several key nodes that handle different aspects of the workflow:

1. **triage_router**: This node is responsible for analyzing incoming emails and classifying them into three categories:
   - **RESPOND**: Emails that require a response from the assistant.
   - **NOTIFY**: Important emails that don't need a response but should be brought to the user's attention.
   - **IGNORE**: Low-priority emails that can be safely ignored.
   
   The router uses a structured output LLM to make this classification and then routes to the appropriate next node based on its decision.

2. **triage_interrupt_handler**: When an email is classified as "notify," this handler creates an interrupt to display the email in Agent Inbox and collect human feedback. This allows users to:
   - Confirm the notification classification
   - Provide feedback on how they would prefer similar emails to be classified in the future
   
   This feedback loop is crucial for improving the assistant's classification over time.

3. **llm_call**: This node invokes the LLM with the available tools to decide how to respond to an email. The LLM might decide to:
   - Draft a response email
   - Schedule a meeting
   - Ask the user a question
   - Or mark the email as done

Each of these nodes plays a specific role in the overall email processing workflow.

```python

# Nodes 
def triage_router(state: State) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
        
    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
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

def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl.format(background=default_background,
                                                                                  response_preferences=default_response_preferences, 
                                                                                  cal_preferences=default_cal_preferences)}
                ]
                + state["messages"]
            )
        ]
    }
```

### The interrupt_handler: Human Review for Tool Execution

The `interrupt_handler` is the core HITL component of our response agent. Its job is to examine the tool calls that the LLM wants to make and determine which ones need human review before execution. Here's how it works:

1. **Tool Selection**: The handler maintains a list of "HITL tools" that require human approval:
   - `write_email`: Since sending emails has significant external impact
   - `schedule_meeting`: Since scheduling meetings affects calendars
   - `Question`: Since asking users questions requires direct interaction

2. **Direct Execution**: Tools not in the HITL list (like `check_calendar_availability`) are executed immediately without interruption. This allows low-risk operations to proceed automatically.

3. **Context Preparation**: For tools requiring review, the handler:
   - Retrieves the original email for context
   - Formats the tool call details for clear display
   - Configures which interaction types are allowed for each tool type

4. **Interrupt Creation**: The handler creates a structured interrupt request with:
   - The action name and arguments
   - Configuration for allowed interaction types
   - A description that includes both the original email and the proposed action

5. **Response Processing**: After the interrupt, the handler processes the human response:
   - **Accept**: Executes the tool with original arguments
   - **Edit**: Updates the tool call with edited arguments and then executes
   - **Ignore**: Cancels the tool execution
   - **Response**: Records feedback without execution

This handler ensures humans have oversight of all significant actions while allowing routine operations to proceed automatically. The ability to edit tool arguments (like email content or meeting details) gives users precise control over the assistant's actions.

```python

def interrupt_handler(state: State):
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
                        
        elif response["type"] == "edit":

            # Tool selection 
            tool = tools_by_name[tool_call["name"]]
            
            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Save feedback in memory and update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":
                
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
            # Don't execute the tool
            result.append({"role": "tool", "content": "Tool execution cancelled by user", "tool_call_id": tool_call["id"]})
            
        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            result.append({"role": "tool", "content": f"Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            
    return {"messages": result}
```

### The Complete HITL Email Assistant Workflow

Now we can integrate everything into a complete workflow that connects all the components. The workflow consists of two main parts:

1. **Response Agent Subgraph**:
   First, we build a standalone agent that can handle email responses:
   - The `llm_call` node generates responses or tool calls
   - The `should_continue` function checks if the agent is done or needs to use a tool
   - The `interrupt_handler` manages human review of tool execution
   - The cycle continues until the agent reaches a conclusion
   
   This response agent is compiled as a reusable subgraph.

2. **Overall Email Assistant Workflow**:
   Then, we create the main workflow that:
   - Starts with `triage_router` to classify the email
   - Routes to `triage_interrupt_handler` for NOTIFY classifications
   - Routes to `response_agent` for RESPOND classifications
   - Ends immediately for IGNORE classifications

This architecture provides a clean separation of concerns, with distinct components for triage, response generation, and human oversight. The resulting workflow gives us a complete email assistant that:

- Analyzes incoming emails
- Correctly routes them based on importance
- Engages humans for oversight at critical decision points
- Responds appropriately to important emails

The final graph visualization shows the complete flow from email input through triage and, when necessary, through the response generation process with human oversight at each significant step.

```python

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

# Build overall workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    
)

email_assistant = overall_workflow.compile()
display(Image(email_assistant.get_graph().draw_mermaid_png()))
```

### Feedback

```python
import uuid 
from langgraph.checkpoint.memory import MemorySaver
from src.email_assistant.email_assistant_hitl import overall_workflow

# Respond
email_input =  {
    "to": "Lance Martin <lance@company.com>",
    "author": "Project Manager <pm@client.com>",
    "subject": "Tax season let's schedule call",
    "email_thread": "Lance,\n\nIt's tax season again, and I wanted to schedule a call to discuss your tax planning strategies for this year. I have some suggestions that could potentially save you money.\n\nAre you available sometime next week? Tuesday or Thursday afternoon would work best for me, for about 45 minutes.\n\nRegards,\nProject Manager"
}

# Compile the graph
checkpointer = MemorySaver()
graph = overall_workflow.compile(checkpointer=checkpointer)
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# Run the graph until the first interrupt
for chunk in graph.stream({"email_input": email_input}, config=thread_config):
   print(chunk)
```

```python
from langgraph.types import Command
# response = adds FEEDBACK for future reference, which is not use yet! We need memory to use it.
for chunk in graph.stream(Command(resume=[{"type": "response", 
                                          "args": "Let's suggest 30 minute calls in the future!'"}]), config=thread_config):
   print(chunk)
```

```python
Interrupt_Object = chunk['__interrupt__'][0]
Interrupt_Object.value[0]['action_request']
``` 

```python
from langgraph.types import Command
# Accept the email to send
for chunk in graph.stream(Command(resume=[{"type": "accept", 
                                          "args": ""}]), config=thread_config):
   print(chunk)
```

```python
state = graph.get_state(thread_config)
for m in state.values['messages']:
    m.pretty_print()
```

### Edit

```python
# Compile the graph
checkpointer = MemorySaver()
graph = overall_workflow.compile(checkpointer=checkpointer)
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# Run the graph until the first interrupt
for chunk in graph.stream({"email_input": email_input}, config=thread_config):
   print(chunk)
```

```python
# edit = edits the tool call ('action': 'schedule_meeting')
for chunk in graph.stream(Command(resume=[{"type": "edit",  
                                           "args": {"args": {"attendees": ['pm@client.com', 'lance@company.com'],
                                                             "subject": "Tax Planning Strategies Discussion",
                                                             "duration_minutes": 30,
                                                             'preferred_day': '2023-11-07'}
                                                             }
                                                             }]), config=thread_config):
   print(chunk)
```

```python
# Accept the email to send
for chunk in graph.stream(Command(resume=[{"type": "accept", 
                                          "args": ""}]), config=thread_config):
   print(chunk)
```


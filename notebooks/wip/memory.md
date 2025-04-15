# Memory

We've used Human-in-the-Loop (HITL) to allow users to review, provide feedback on, or even correct the assistant's decisions. This is great, but it would be even better if the assistant *could learn from* the user's feedback and adapt to their preferences over time. This is where memory comes in. Memory is a critical and emerging component of agent systems, allowing them to learn and improve over time. In this section, we'll add a memory component to our email assistant, allowing it to learn from user feedback and adapt to their preferences over time. This gives us more confidence that the assistant acts on our behalf with personalization. 

![overview-img](img/overview_memory.png)

## Memory in LangGraph

### Thread-Scoped and Across-Thread Memory

First, it's worth explaining how [memory works in LangGraph](https://langchain-ai.github.io/langgraph/concepts/memory/). LangGraph offers two distinct types of memory that serve complementary purposes in agent systems:

**Thread-Scoped Memory (Short-term)** operates within the boundaries of a single conversation thread. It's automatically managed as part of the graph's state and persisted through thread-scoped checkpoints. This memory type retains conversation history, uploaded files, retrieved documents, and other artifacts generated during the interaction. Think of it as the working memory that maintains context within one specific conversation, allowing the agent to reference earlier messages or actions without starting from scratch each time.

**Across-Thread Memory (Long-term)** extends beyond individual conversations, creating a persistent knowledge base that spans multiple sessions. This memory is stored as JSON documents in a memory store, organized by namespaces (like folders) and distinct keys (like filenames). Unlike thread-scoped memory, this information persists even after conversations end, enabling the system to recall user preferences, past decisions, and accumulated knowledge. This is what allows an agent to truly learn and adapt over time, rather than treating each interaction as isolated.

The [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) is the foundation of this architecture, providing a flexible database where memories can be organized, retrieved, and updated. What makes this approach powerful is that regardless of which memory type you're working with, the same Store interface provides consistent access patterns. This allows your agent's code to remain unchanged whether you're using a simple in-memory implementation during development or a production-grade database in deployment. 

### LangGraph Store

LangGraph offers different [Store implementations depending on your deployment scenario](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore):

1. **Pure In-Memory (e.g., notebooks)**:
   - Uses `from langgraph.store.memory import InMemoryStore`
   - Purely a Python dictionary in memory with no persistence
   - Data is lost when the process terminates
   - Useful for quick experiments and testing
   - Includes semantic search with cosine similarity

2. **Local Development with `langgraph dev`**:
   - Similar to InMemoryStore but with pseudo-persistence
   - Data is pickled to the local filesystem between restarts
   - Lightweight and fast, no need for external databases
   - Semantic search uses cosine similarity for embedding comparisons
   - Great for development but not designed for production use

3. **LangGraph Platform or Production Deployments**:
   - Uses PostgreSQL with pgvector for production-grade persistence
   - Fully persistent data storage with reliable backups
   - Scalable for larger datasets
   - High-performance semantic search via pgvector
   - Default distance metric is cosine similarity (customizable)

Let's use the `InMemoryStore` here in the notebook! 

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

Memories are namespaced by a tuple, which in this specific example will be (`<user_id>`, "memories"). The namespace can be any length and represent anything, does not have to be user specific.

```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

We use the `store.put` method to save memories to our namespace in the store. When we do this, we specify the namespace, as defined above, and a key-value pair for the memory: the key is simply a unique identifier for the memory (memory_id) and the value (a dictionary) is the memory itself.

```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
```

We can read out memories in our namespace using the `store.search` method, which will return all memories for a given user as a list. The most recent memory is the last in the list. Each memory type is a Python class (`Item`) with certain attributes. We can access it as a dictionary by converting via `.dict` as above. The attributes it has are shown below, but the most important ones is typically `value`.

```python
memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()
```

To use this in a graph, all we need to do is compile the graph with the store:

```
# We need this because we want to enable threads (conversations)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
# We need this because we want to enable across-thread memory
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
# Compile the graph with the checkpointer and store
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```

## Memory in LangGraph

Let's take our graph used with HITL and add memory to it.

```python
%cd ..
%load_ext autoreload
%autoreload 2
```

Here we set up the triage router node, which is the first node in our graph.

```python
from typing import Literal
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl_memory, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
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

Now, this is the critical part! Right now we log feedback :
```
Here is feedback on how the user would prefer the email to be classified
```

### Memory Management with LangMem

But we don't do anything with it! Let's change that by simply adding the feedback to the memory. What we *want* to do is fairly straightforward: we want to add the feedback to the memory `Store`. If we compile our graph with the store, we can access the store in any node. So that is not a problem! But we have to answer two questions: 1) how do we want the memory to be structured? 2) how do we want to update the memory? 

This is where [LangMem](https://langchain-ai.github.io/langmem/) comes in! LangMem is a lightweight library that can be used on top of the LangGraph Store to provide a more user-friendly API for memory management. We can create a `create_memory_store_manager` that takes care of a few nice things: 

1) it will create a namespace in the `Store` for us 
2) it will initialize the store with our default instructions (default_triage_instructions)
3) it allows us to specify if we want a collection `enable_inserts=True` of memories
4) it allows us to specify if we just want to update one memory "profile" `enable_inserts=False`
5) it handles updating the memory based upon input messages

```python
from langmem import create_memory_store_manager
from email_assistant.prompts import default_triage_instructions

# Feedback memory managers for writing to memory Store 
triage_feedback_memory_manager = create_memory_store_manager(
    init_chat_model("openai:gpt-4o", temperature=0.0),
    namespace=("email_assistant", "triage_preferences"),
    instructions="""Extract user email triage preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False, # Update profile in-place,
    enable_deletes=False # Do not delete profile from memory
    default=default_triage_instructions
)
```

Now we can used this in our by directly calling the `invoke` method.

```python
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager

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
        # Update memory with feedback using the memory manager
        triage_feedback_memory_manager.invoke({"messages": messages})
        goto = END

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)
```

We can create a memory manager for each memory type we want to store. 

Each memory manager is specialized for a specific type of information the assistant needs to remember. Let's examine how we set up different memory types for our email assistant:

1. **Response Preferences Manager**: This memory manager captures and maintains user preferences for email responses, such as tone, style, and formatting preferences. It uses a "profile" approach (with `enable_inserts=False`) to maintain a single, consolidated set of preferences that get updated over time, rather than creating many discrete memories.

2. **Calendar Preferences Manager**: Similar to response preferences, this manager stores scheduling preferences like preferred meeting durations, times of day, and days of the week. It's also implemented as a profile for consistent reference.

3. **Background Knowledge Manager**: Unlike the preference managers, this one uses a "collection" approach (with `enable_inserts=True`) to accumulate discrete facts about people, projects, and contexts. Each new piece of relevant information becomes a separate memory entry that can be retrieved based on relevance.

The key distinction is in how these memories are updated:
- Profiles (response and calendar preferences) consolidate feedback into a single document that gets refined over time
- Collections (background knowledge) grow by adding new discrete memories while potentially removing outdated ones

```python
response_preferences_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "response_preferences"),
    instructions="""You goal is to maintain a profile that contains a user's email response preferences. 
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
```

### Accessing Memory in the Triage Router

The triage router now leverages stored memory to make more personalized classification decisions. Let's see how memory transforms this function:

```python
def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    
    # Search for existing triage_preferences memory
    results = triage_feedback_memory_manager.search()
    triage_instructions=results[0].value['content']
    
    # Search for existing background memory
    results = background_memory_manager.search()
    # Handle collection of memory objects
    memories = []
    for result in results:
        memories.append(result.value['content']['content'])
    background_content = "\n".join(memories)
        
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
```

### Incorporating Memory into LLM Responses

Now that we have memory managers set up, we need to use the stored preferences when generating responses. The `llm_call` function demonstrates how to retrieve and incorporate memory into the LLM's context:

```python
def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not"""

    # Search for existing cal_preferences memory
    results = cal_preferences_memory_manager.search()
    cal_preferences=results[0].value['content']['content']
    
    # Search for existing response_preferences memory
    results = response_preferences_memory_manager.search()
    response_preferences=results[0].value['content']['content']

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl_memory.format(response_preferences=response_preferences, 
                                                                                         cal_preferences=cal_preferences)}
                ]
                + state["messages"]
            )
        ]
    }
```

### Memory Integration in the Interrupt Handler

The interrupt handler is where memory truly shines, as it's responsible for capturing user feedback and using it to update our various memory stores. This function showcases how we:

1. **Process User Feedback**: When a user edits an email response or provides feedback, we capture that information
2. **Update Relevant Memory**: We route the feedback to the appropriate memory manager based on the context
3. **Learn Continuously**: Each interaction becomes a learning opportunity for the system

Let's break down the key memory interactions:
    
```python
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
```

This is the same as before. 

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
```

Now, we can compile the graph with the store.
```python
import uuid 
from langgraph.checkpoint.memory import MemorySaver
from src.email_assistant.email_assistant_hitl_memory import overall_workflow
checkpointer = MemorySaver()
store = InMemoryStore()

# Respond
email_input =  {
    "to": "Lance Martin <lance@company.com>",
    "author": "Project Manager <pm@client.com>",
    "subject": "Tax season let's schedule call",
    "email_thread": "Lance,\n\nIt's tax season again, and I wanted to schedule a call to discuss your tax planning strategies for this year. I have some suggestions that could potentially save you money.\n\nAre you available sometime next week? Tuesday or Thursday afternoon would work best for me, for about 45 minutes.\n\nRegards,\nProject Manager"
}

# Compile the graph
graph = overall_workflow.compile(checkpointer=checkpointer, store=store)
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# Run the graph until the first interrupt
for chunk in graph.stream({"email_input": email_input}, config=thread_config):
   print(chunk)
```

We can see the initial preferences.
```python
# We can see the initial response preferences
results = store.search(("email_assistant", "triage_preferences"))
results
```

And:
```python
# We can see the initial response preferences
results = store.search(("email_assistant", "cal_preferences"))
results[0].value['content']
```

Add feedback:
```python
from langgraph.types import Command
# response = adds FEEDBACK for future reference, which is not use yet! We need memory to use it.
for chunk in graph.stream(Command(resume=[{"type": "response", 
                                          "args": "Always use 30 minute calls in the future!'"}]), config=thread_config):
   print(chunk)
```

Updated preferences:
```python
# We can see the updated response preferences
results = store.search(("email_assistant", "cal_preferences"))
results[0].value['content']
```

Then, accept the calendar invite:
```python
# Accept the invite
for chunk in graph.stream(Command(resume=[{"type": "accept", 
                                          "args": ""}]), config=thread_config):
   print(chunk)
```

And accept the email to send:
```python
# Accept the email to send
for chunk in graph.stream(Command(resume=[{"type": "accept", 
                                          "args": ""}]), config=thread_config):
   print(chunk)
```






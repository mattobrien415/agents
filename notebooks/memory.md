# Personalized Email Assistant with Memory

In this final enhancement of our email assistant, we add a powerful memory component that enables the assistant to learn from user feedback and adapt to personal preferences over time. This memory-enabled system creates a truly personalized experience, allowing the assistant to become increasingly effective with each interaction.

![overview-img](img/overview_memory.png)

## Memory Architecture in LangGraph

Our email assistant leverages [LangGraph's store mechanism](https://langchain-ai.github.io/langgraph/reference/store/) and [LangMem](https://langchain-ai.github.io/langmem/) to implement a robust memory system. This enables the assistant to persist information across sessions and provide increasingly personalized responses.

![memory-img](img/memory.png)

### The LangGraph Store: Persistent Memory Database

At the core of our memory system is the LangGraph Store, a persistent database that retains information across sessions:

```python
from langgraph.store.base import BaseStore
```

The Store is a hierarchical key-value database with several important features:

1. **Persistence**: Data remains available between conversations and even after restarts, unlike in-memory state
2. **Namespaced Organization**: Data is organized in tuples like `("email_assistant", "response_preferences")`, allowing structured storage of different memory types
3. **Automatic Availability**: When running with `langgraph dev` or on LangGraph Platform, the Store is automatically configured
4. **Search Capabilities**: Supports both exact key lookup and semantic search (using embeddings)
5. **Integration with Nodes**: Every node function can access the Store via a parameter:

```python
def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    # Access memory from store
    results = store.search(("email_assistant", "triage_preferences"))
    # ...
```

This store-based architecture allows our assistant to maintain a growing knowledge base across conversations, continuously learning from each interaction.

### Core Memory Capabilities

The memory system built on the Store provides four essential capabilities:

1. **Persistent Storage**: Using LangGraph's Store, memories are preserved between sessions
2. **Memory Categories**: Different types of memories are organized in dedicated namespaces
3. **Semantic Search**: Memories can be retrieved based on relevance to the current context
4. **Feedback Learning**: The system continuously improves by capturing user feedback

## Specialized Memory Types with LangMem

Our assistant uses four specialized memory types implemented with [LangMem](https://langchain-ai.github.io/langmem/), each dedicated to a particular aspect of email processing:

1. **Triage Preferences**: Rules for classifying emails (ignore, respond, notify)
2. **Response Preferences**: Style, tone, and content preferences for email responses
3. **Calendar Preferences**: Scheduling preferences for meetings
4. **Background Knowledge**: Contextual information about colleagues, projects, etc.

### Memory Access: create_search_memory_tool

For each memory type, we create a specialized search tool using `create_search_memory_tool`:

```python
# Create a tool to search response preferences memories
response_preferences_tool = create_search_memory_tool(
    namespace=("email_assistant", "response_preferences"), 
    name="response_preferences"
)
```

This creates a LangChain tool the agent can use to search through stored memories. Let's break down how this works:

- **namespace**: The memory location in our store, using a tuple structure for hierarchical organization
- **name**: The name of the tool, which the agent will use when deciding to search memories

When the agent needs to recall response preferences, it can call this tool like this:

```python
# Example of how the agent would use the memory search tool
response_preferences = response_preferences_tool.invoke({"query": "How should I respond to meeting requests?"})
```

The tool performs a semantic search in the specified namespace and returns relevant memories to the agent, which can then incorporate these preferences into its decision-making.

The agent has access to all these memory search tools:

```python
# All tools available to the agent
tools = [
    write_email, 
    schedule_meeting, 
    check_calendar_availability, 
    Question, 
    response_preferences_tool,  # Search response style preferences
    cal_preferences_tool,       # Search calendar scheduling preferences
    background_tool,            # Search background knowledge
    triage_preferences_tool,    # Search email classification rules
    Done
]
```

## Memory Managers for Writing to Memory

To update memories based on feedback, we use memory managers created with `create_memory_store_manager` that process user interactions and extract relevant information:

```python
# Create a memory manager for response preferences
response_preferences_memory_manager = create_memory_store_manager(
    llm,  # The LLM that will process and extract memories
    namespace=("email_assistant", "response_preferences"),  # Where to store memories
    instructions="""Extract user email response preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False,  # Update the existing memory rather than adding new entries
    enable_deletes=False   # Don't allow deletion of existing memories
)
```

Let's examine how this memory manager works:

1. **Processing Feedback**: The memory manager takes conversation messages that contain feedback about email responses
2. **LLM Analysis**: The LLM analyzes these messages to extract meaningful preferences
3. **Memory Consolidation**: Rather than storing each piece of feedback separately, it updates a consolidated set of rules
4. **Storage**: The extracted preferences are saved in the specified namespace in the store

For example, when a user edits an email response, we capture that feedback and update our response preferences:

```python
# When a user edits an email response
if response["type"] == "edit" and tool_call["name"] == "write_email":
    # Extract edited content
    edited_args = response["args"]["args"]
    
    # Update response preferences memory with the edited version
    response_preferences_memory_manager.invoke({
        "messages": state["messages"] + [
            {"role": "user", "content": f"Here is a better way to respond to emails: {edited_args}"}
        ]
    })
    
    # Continue with executing the tool with edited args...
```

This approach allows our system to learn continually from user feedback, gradually building a comprehensive understanding of preferences.

## Learning From Human Feedback

The truly powerful aspect of our memory system is how it learns from human feedback through Agent Inbox interrupts. Here's how that works for triage classification:

```python
def triage_interrupt_handler(state: State, store: BaseStore) -> Command[Literal["response_agent", "__end__"]]:
    # ...
    
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
    # ...
```

When a user provides feedback, it's processed by the appropriate memory manager, which extracts useful information and updates the memory store.

Similarly, when a user edits a proposed email response:

```python
# Save feedback in memory and update the write_email tool call with the edited content from Agent Inbox
if tool_call["name"] == "write_email":
    # Let's save our response preferences in memory
    response_preferences_memory_manager.invoke({
        "messages": state["messages"] + [{"role": "user", "content": f"Here is a better way to respond to emails: {edited_args}"}]
    })
    
    # Update the AI message's tool call with edited content...
```

The changes are not just applied to the current email but also used to update the response preferences memory for future interactions.

## Background Knowledge Accumulation

Unlike the other memory types that are focused on preferences, the background memory accumulates factual information:

```python
background_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "background"),
    instructions="""Extract user email background information about the user, their key connections, and other relevant information.
    Format this as a collection of short memories that can be easily recalled.""",
    enable_inserts=True, # Update background in-place,
    enable_deletes=True # Since this is a collection, we can delete items (if they are no longer relevant)
)
```

This type of memory grows over time as the assistant processes more emails, building a richer understanding of the user's context:

```python
# Remember facts from the conversation with background memory manager
background_memory_manager.invoke({"messages": state["messages"] + result})
```

## Memory Initialization

When the assistant is first used, default values are loaded for each memory type:

```python
# Calendar preferences
results = store.search(
    ("email_assistant", "cal_preferences")
)
if not results:
    cal_preferences = default_cal_preferences
    cal_preferences_memory_manager.invoke({"messages": [{"role": "user", "content": cal_preferences}]})
```

These defaults provide a starting point, which is then refined through user interactions.

## Testing with Local Deployment

To test our memory-enabled assistant, we'll use the LangGraph local server which automatically sets up a persistent Store for us:

1. Run `langgraph dev` to start the LangGraph server locally
2. Use the configuration in `langgraph.json` that points to our memory-enabled assistant:
   ```json
   "graphs": {
       "email_assistant": "./src/email_assistant/email_assistant_hitl_memory.py:email_assistant",
   }
   ```
3. Connect to Agent Inbox as we did for the HITL version
4. Send multiple emails to observe how the assistant learns from feedback

In the LangGraph Studio, we can view the memories accumulated in the "Memory" store section. This visualization shows exactly what's being stored in each memory namespace, allowing us to inspect how the assistant is learning from our feedback over time.

## Benefits of Memory-Enabled Assistants

1. **Personalization**: The assistant adapts to individual preferences rather than using generic rules
2. **Continuous Learning**: Each interaction improves future performance through feedback
3. **Context Awareness**: Growing background knowledge makes responses more relevant
4. **Efficiency**: Users don't have to repeat the same preferences or corrections
5. **Trust Building**: As the assistant becomes more aligned with user preferences, trust increases

By combining human-in-the-loop feedback with persistent memory, we've created an assistant that truly learns and improves over time, offering increasingly personalized email assistance.
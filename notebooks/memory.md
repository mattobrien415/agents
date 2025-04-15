# Memory (WIP)

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

#### Store Implementations in Different Deployment Scenarios

LangGraph offers different Store implementations depending on your deployment scenario:

1. **Pure In-Memory (e.g., notebooks, tests)**:
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

The amazing thing about LangGraph's Store abstraction is that your code remains the same regardless of which implementation is used - the differences are handled transparently by the framework.

This store-based architecture allows our assistant to maintain a growing knowledge base across conversations, continuously learning from each interaction.

### Memory Types: Profiles vs. Collections

LangGraph supports two fundamental approaches to memory:

#### Memory Profiles

Profiles are comprehensive, unified documents about a specific topic that get updated over time:

```python
# Storing a preferences profile directly using the Store API
store.put(
    ("email_assistant", "response_preferences"),  # Namespace
    "preferences",                                # Key
    {                                             # Value
        "content": {
            "content": "Always be professional. Use formal greetings.",
            "metadata": {"source": "user_feedback"}
        }
    }
)

# Retrieving a profile
profile = store.get(("email_assistant", "response_preferences"), "preferences")
preferences = profile.value["content"]["content"]
```

In our email assistant, we use profiles for:
- **Triage Preferences**: Rules for email classification
- **Response Preferences**: How emails should be formatted and styled  
- **Calendar Preferences**: User scheduling preferences

Profiles are ideal when you need a consistent, consolidated view of preferences or rules.

#### Memory Collections

Collections are groups of discrete memories that grow over time, each capturing a specific piece of information:

```python
# Adding an item to a memory collection
store.put(
    ("email_assistant", "background"),  # Namespace
    f"fact_{timestamp}",                # Unique key for this memory
    {                                   # Value
        "content": {
            "content": "John is allergic to peanuts.",
            "metadata": {"source": "email_conversation"}
        }
    }
)

# Retrieving all memories in a collection
memories = store.search(("email_assistant", "background"))
all_facts = "\n".join([memory.value["content"]["content"] for memory in memories])
```

In our assistant, we use a collection for:
- **Background Knowledge**: Facts about colleagues, projects, and contexts that accumulate over time

Collections are useful when information naturally consists of discrete facts that grow and evolve over time.

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

## Working with Memory Using LangMem

While we could use the LangGraph Store API directly, [LangMem](https://langchain-ai.github.io/langmem/) provides powerful abstractions that simplify memory management. LangMem integrates seamlessly with LangGraph's Store to provide more sophisticated memory capabilities.

### Memory Store Managers for Writing to Memory

The core of our memory system is based on `create_memory_store_manager`, which handles the complex process of extracting, organizing, and storing memories:

```python
# Create a memory manager for response preferences (a profile)
response_preferences_memory_manager = create_memory_store_manager(
    llm,  # The LLM that will process and extract memories
    namespace=("email_assistant", "response_preferences"),  # Where to store memories
    instructions="""Extract user email response preferences into a single set of rules.
    Format the information as a string explaining the criteria for each category.""",
    enable_inserts=False,  # Update the existing profile rather than adding new entries
    enable_deletes=False   # Don't allow deletion of existing memories
)

# Create a memory manager for background knowledge (a collection)
background_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "background"),
    instructions="""Extract user email background information about the user, their key connections, and other relevant information.
    Format this as a collection of short memories that can be easily recalled.""",
    enable_inserts=True,  # Allow adding new entries to the collection
    enable_deletes=True   # Allow removing outdated entries
)
```

These managers handle different memory types differently:

1. **For Profiles** (like response_preferences): The manager updates a single, consolidated document
2. **For Collections** (like background): The manager can add new discrete memories over time

### Memory Manager Workflow

Let's examine how memory managers work:

1. **Message Analysis**: The memory manager processes conversation messages:
   ```python
   # Provide conversation context to extract memories from
   response_preferences_memory_manager.invoke({
       "messages": [
           {"role": "user", "content": "Please use more formal language in emails."}
       ]
   })
   ```

2. **LLM-Based Extraction**: The LLM extracts meaningful information from the messages

3. **Intelligent Storage**:
   - For profiles: Updates the single document with consolidated information
   - For collections: May add new discrete memories to the collection

4. **Store Integration**: All changes are automatically persisted in the LangGraph Store

For example, when a user edits an email response, we capture that feedback and update our response preferences profile:

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
```

Similarly, when we detect potentially useful background information:

```python
# Remember facts from a conversation
background_memory_manager.invoke({
    "messages": [
        {"role": "user", "content": "My colleague John is allergic to peanuts."}
    ]
})
```

This approach allows our system to learn continually from user feedback and conversations, building both consolidated preference profiles and growing collections of background knowledge.

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

Since background memory is structured as a collection of entries rather than a single profile, we use a specialized utility function to retrieve it:

```python
def get_memory_collection(store, namespace, default_content, memory_manager=None):
    """Get a collection of memories from the store or initialize with default if it doesn't exist."""
    # Search for existing memories
    results = store.search(namespace)
    
    # If memories exist, concatenate all their contents
    if results:
        # Handle collection of memory objects
        memories = []
        for result in results:
            memories.append(result.value['content']['content'])
        return "\n".join(memories)
    
    # If no memories exist and we have a manager, initialize with default
    if not results and memory_manager:
        memory_manager.invoke({"messages": [{"role": "user", "content": default_content}]})
    
    # Return the default content
    return default_content
```

And we can use it like this:

```python
# Get background information (combining all entries)
background_content = get_memory_collection(
    store,
    ("email_assistant", "background"),
    default_background,
    background_memory_manager
)
```

## Utility Functions for Memory Access

To simplify memory access and initialization in our application, we've created utility functions that handle both types of memories:

### Memory Profile Initialization

For memory profiles (like preferences), we've created a utility function:

```python
def get_memory_profile(store, namespace, default_content, memory_manager=None):
    """Get memory profile from the store or initialize with default if it doesn't exist."""
    # Search for existing memory
    results = store.search(namespace)
    
    # If memory exists, return its content
    if results:
        return results[0].value['content']['content']
    
    # If memory doesn't exist and we have a manager, initialize it
    if not results and memory_manager:
        memory_manager.invoke({"messages": [{"role": "user", "content": default_content}]})
    
    # Return the default content
    return default_content
```

This utility makes it simple to retrieve and initialize memory profiles:

```python
# Get triage preferences from memory or initialize with defaults
triage_instructions = get_memory_profile(
    store, 
    ("email_assistant", "triage_preferences"), 
    default_triage_instructions,
    triage_feedback_memory_manager
)
```

### Memory Collection Initialization

For memory collections (like background knowledge), we have a separate utility:

```python
def get_memory_collection(store, namespace, default_content, memory_manager=None):
    """Get a collection of memories from the store or initialize with default if it doesn't exist."""
    # Search for existing memories
    results = store.search(namespace)
    
    # If memories exist, concatenate all their contents
    if results:
        # Handle collection of memory objects
        memories = []
        for result in results:
            memories.append(result.value['content']['content'])
        return "\n".join(memories)
    
    # If no memories exist and we have a manager, initialize with default
    if not results and memory_manager:
        memory_manager.invoke({"messages": [{"role": "user", "content": default_content}]})
    
    # Return the default content
    return default_content
```

We use this to handle background knowledge:

```python
# Get background information from memory or initialize with defaults
background_content = get_memory_collection(
    store,
    ("email_assistant", "background"),
    default_background,
    background_memory_manager
)
```

Together, these utilities provide a clean interface for working with both memory profiles and collections, handling initialization automatically when needed.

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



### Memory Architecture

Each memory type is updated through specialized components, which can take in messages and update the memory. 

```
triage_feedback_memory_manager = create_memory_store_manager(
    llm,
    namespace=("email_assistant", "triage_preferences"),
    instructions="Extract user email triage preferences into a single set of rules"
)
```

The system also includes specialized tools to search these memories, which the agent can use to fetch memories when needed:

```
response_preferences_tool = create_search_memory_tool(
    namespace=("email_assistant", "response_preferences")
)
```

### Learning From Feedback

The assistant updates its memory in several key scenarios:

#### 1. Triage Classification Feedback

When the assistant classifies an email as "notify" and sends it to Agent Inbox, you have options:

* **Accept**: The classification stands without memory updates
* **Provide Feedback**: Your feedback (e.g., "This should be classified as 'respond'") is processed by the `triage_feedback_memory_manager`, which extracts rules and updates the triage preferences memory

This gradually improves the assistant's understanding of which emails should fall into each category.

#### 2. Email Response Editing

When reviewing a draft email response in Agent Inbox:

* **Edit Mode**: When you edit the content of a response, the assistant doesn't just use your edits for that specific email - it also updates its `response_preferences_memory` with insights from your changes
* **Feedback Mode**: Any comments you provide about the response style, tone, or content are stored as response guidelines

For example, if you consistently edit greetings from "Hello" to "Hi" or add more technical details to responses, the assistant learns these preferences over time.

#### 3. Meeting Scheduling Feedback

When the assistant proposes scheduling a meeting:

* **Edit Meeting Details**: If you change meeting duration, preferred days, or other parameters, these updates are stored in `calendar_preferences_memory`  
* **Provide Scheduling Guidelines**: Any direct feedback about scheduling preferences is processed and stored

This might include learning that you prefer morning meetings, specific meeting durations, or avoiding certain days.

#### 4. Background Information Accumulation

The assistant continuously builds its knowledge base about your work context:

* Every message provides potential background information about projects, team members, and organizational context
* This information is automatically extracted and stored in background_memory
* Future interactions then have access to this growing knowledge base

For example, if an email mentions a project deadline or a team member's role, this information becomes available for future reference.

### Memory Benefits

This semantic memory approach offers several key advantages:

1. **Personalization**: The assistant adapts to your specific preferences rather than using generic rules
2. **Efficiency**: You don't need to repeat the same feedback - the assistant learns from each interaction
3. **Context Awareness**: Background memory helps the assistant understand references to projects and people
4. **Continuous Improvement**: The system gets more accurate over time through regular use and feedback

### Running the Memory-Enabled Assistant

To use the memory-enabled version, you can reference the `langgraph.json` file:
```json
"graphs": {
    "email_assistant": "./src/email_assistant/email_assistant_hitl_memory.py:email_assistant",
},
```

This graph uses feedback from HITL (Agent Inbox) to update the memory. 

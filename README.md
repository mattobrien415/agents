# Interrupt Workshop 

AI agents promise to transform how we work, but there's often a gap between hype and reality: to truly act on our behalf, agents need to learn and remember our preferences for personalization. In this repo, we show how to build self-improving and personalized agents from scratch using LangChain, LangGraph, and LangSmith. 

In this hands-on workshop, you'll build an intelligent email assistant that actually learns how you think. Unlike generic AI tools, your assistant will master your unique approach to triaging and responding to emails through a combination of memory and human-in-the-loop feedback. 

![interrupt_conf_high_level](https://github.com/user-attachments/assets/37c4376b-519b-4f53-9525-e924fa067cfd)

## Environment Setup 

### Prerequisites

1. Set up your LangSmith API key as an environment variable:
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key
   ```
2. Set up your OpenAI API key (default LLM used for the Evaluator and E-mail Assistant):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

Create a virtual environment and activate it:
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

## Baseline Assistant 

### Functionality 

An e-mail assistant should do a few things. First, the assistant analyzes incoming emails and classifies them. Here, we will classify them into three categories (respond, notify, ignore). For emails classified as needing a response, the assistant can call tools to check calendar availability, schedule meetings, and draft emails. 

### Tool calling agent vs Workflow 

There are [a few approaches to building an email assistant](https://langchain-ai.github.io/langgraph/tutorials/workflows/), including:

1. **Tool-calling Agent** (`baseline_agent.py`):
   - This is a [baseline tool calling agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/#how-to-use-the-pre-built-react-agent)
   - A single LLM handles *all* tasks using a collection of tools 
   - It has a dedicated triage tool to categorize emails
   - Has tools for to e-mail drafting, calendar scheduling, and calendar search
   - All decision-making happens within one LLM component

2. **Agentic Workflow** (`email_assistant.py`):
   - Includes a clear separation of concerns between triage step and an agent to handle e-mail response
   - Has router node to decide what to do with an email
   - E-mail response is handled by a separate agent with tools for e-mail drafting, calendar scheduling, and calendar search

Agentic workflows offload some tools from the agent to a dedicated router. There are a few reasons to do this. First, we know that we always want the workflow to start with a triage step. So, we can break out that step from the agent and have a dedicated router to handle it. This allows the agent to focus only on responding to e-mails. Second, with routing offloaded we can more comfortably add tools to the agent, such as tool related to memory. 

### Run E-mail Assistants 

Install uv package manager and start LangGraph Platform server locally:
```shell
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

Here we can see the two assistants easily and test them. You will see in the `langgrah.json` that we point to the compiled graphs that we want to load.
```shell
  "graphs": {
      "email_assistant": "./src/email_assistant/email_assistant.py:email_assistant",
      "baseline_agent": "./src/email_assistant/baseline_agent.py:email_assistant"
    },
```

![Screenshot 2025-04-04 at 4 06 18 PM](https://github.com/user-attachments/assets/72f21b12-c708-4fca-a9c6-8fc3faa2ef82)

In studio, you can test both assistants with some email inputs directly to see what the assistants will do. Try some of the example emails in `eval/email_dataset.py` to see the assistants in action (e.g., `STANDARD_EMAIL` or `NOTIFICATION_EMAIL`).

## Evaluation

### Structure 

This uses LangSmith to perform evaluations in two different ways: it shows how to create a LangSmith dataset and run evaluations on this dataset and it shows how to run tests across all agents using Pytest.

1. **Dataset**: A collection of sample emails with ground truth classifications, responses, and expected tool calls is defined in `eval/email_dataset.py`

2. **Automated Testing** (`tests/test_email_assistant.py`):
   - Uses [Pytest with LangSmith to test](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) all email assistant implementations
   - Tests responses against specific criteria using LLM-as-judge
   - Verifies correct tool calls against expected tool calls defined in `eval/email_dataset.py`
   - Logs results to LangSmith

3. **Email Triage Evaluation** (`eval/evaluate_triage.py`):
   - Uses [LangSmith](https://docs.smith.langchain.com/evaluation) to run and track evaluations
   - Creates a dataset of test emails if it doesn't exist
   - Runs the `baseline tool calling` and `agentic workflow` assistant implementations against the dataset
   - Uses direct string matching to evaluate if classifications match expected values
   - Scores assistants on a 0-1 scale based on exact matches
   - Logs results to LangSmith

4. **Visualization and Reporting**:
   - LangSmith dashboard provides comprehensive analysis
   - Individual test cases appear as separate traces for detailed review

### Run Evaluation 

#### Prerequisites
1. Set up your LangSmith API key as an environment variable:
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key
   ```

#### Run Automated Testing

Use the `run_all_tests.py` script:
```bash
# Run with rich output display
python run_all_tests.py --rich-output

# Run with specific experiment name prefix
python run_all_tests.py --experiment-name "April8Fix"
```

When we run the script, it runs these tests:
```bash
python -m pytest tests/test_response.py -v
python -m pytest tests/test_hitl.py -v
python -m pytest tests/test_memory.py -v
```

These modules test different aspects of functionality:
- `test_response.py` - Tests response generation and tool calling
- `test_hitl.py` - Tests human-in-the-loop functionality
- `test_memory.py` - Tests memory updates and preference learning

Each test is marked with `@pytest.mark.langsmith` to log inputs, outputs, and results to LangSmith, providing detailed traces for analysis. Tests are parametrized to run against the test cases defined in `email_dataset.py`. Each test case is individually logged to LangSmith, allowing detailed performance analysis: 

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to the "Datasets" section to find the results 
3. The dataset name is defined in `run_all_tests.py` as `"E-Mail Assistant Testing: Interrupt Conference"`

![Screenshot 2025-04-08 at 8 07 48 PM](https://github.com/user-attachments/assets/f04a0beb-7c5e-4fec-8e6b-af181ec21300)

#### Run E-mail Triage Evaluation
Use the `evaluate_triage.py` script:
```bash
python -m eval.evaluate_triage
```

This script creates a LangSmith dataset with test emails defined in `eval/email_dataset.py` (if it doesn't exist). It runs `baseline tool calling` and `agentic workflow` assistant implementations against this dataset. The triage evaluation measures how well each assistant correctly classifies emails as "respond", "notify", or "ignore" using direct string matching. After running the evaluations, you can view detailed results in the LangSmith dashboard:

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to the "Datasets" section to find the results 
3. Dataset name in `eval/evaluate_triage.py`: `"Interrupt Workshop: E-mail Triage Dataset"`

![Screenshot 2025-04-01 at 3 04 05 PM](https://github.com/user-attachments/assets/0545212b-4563-4ca8-a748-abe31c84ee18)

## Human-in-the-loop 

### Adding HITL to the Workflow 

What if we want the ability to review and correct the assistant's decisions? We can add a human-in-the-loop (HITL) to the workflow. For this, we will use [Agent Inbox](https://github.com/langchain-ai/agent-inbox) to review and correct the assistant's decisions. To enable human-in-the-loop capabilities, we need to set up a connection between our email assistant and Agent Inbox. We have a graph that is HITL-enabled, so we just need to modify the `langgraph.json` file to point to the HITL-enabled graph.

1. You'll see that the `langgraph.json` file has the HITL-enabled graph: 
```shell
  "graphs": {
      "email_assistant_hitl": "./src/email_assistant/email_assistant_hitl.py:email_assistant",
    },
```

2. Start the LangGraph server:
```shell
$ langgraph dev 
```

3. Open Agent Inbox and connect it to your local server:
```
https://dev.agentinbox.ai/
```

4. Add the new inbox:
   * Graph name: the name from the `langgraph.json` file (`email_assistant_hitl`)
   * Graph URL: `http://127.0.0.1:2024/`

Now you can send email inputs through Studio and interact with them through Agent Inbox.

### Email Assistant HITL Workflow

The HITL-enabled email assistant creates a more interactive experience by involving you at key decision points. Here's how it works:

#### 1. Email Triage Process

When an email arrives, the assistant first analyzes and categorizes it:

* **RESPOND Classification**: The assistant determines the email needs a response. It automatically proceeds to draft a response, which you can review in Agent Inbox before it's sent.

* **IGNORE Classification**: The assistant determines the email can be safely ignored. No human intervention is required, and the workflow ends.

* **NOTIFY Classification**: The assistant determines the email contains important information but doesn't require a response. This classification is sent to Agent Inbox for your review.

#### 2. Agent Inbox Interface

For emails that require human review, you'll see them appear in Agent Inbox with `Required Action` notification:
* The full email content formatted clearly
* The assistant's classification decision
* Action buttons based on the context

![Agent Inbox showing email threads](https://github.com/user-attachments/assets/e45e063b-6e54-49b7-8fef-a0280b52e683)

#### 3. Triage Review Options

When reviewing a NOTIFY classification in Agent Inbox, you have several options:

* **Accept**: You agree with the classification, and the workflow ends
* **Provide Feedback**: You can type a message explaining how you would have classified the email
* **Dismiss**: You can dismiss the notification if it's not important

These options let you confirm or correct the assistant's triage decisions over time.

![Screenshot 2025-04-04 at 3 56 56 PM](https://github.com/user-attachments/assets/718bd6be-410c-4bf7-b59c-0e80b11ff782)

#### 4. Response Review Options

When the assistant drafts an email response (for RESPOND classifications), you can:

* **Edit**: Modify the content of the response directly
* **Accept**: Send the response as drafted
* **Provide Feedback**: Offer guidance on how to improve similar responses
* **Dismiss**: Reject the response entirely

![Screenshot 2025-04-04 at 3 55 57 PM](https://github.com/user-attachments/assets/8a7f4ea1-905a-41de-a5e1-c0c665b0703f)

#### 5. Integration with Workflow

The entire process integrates seamlessly with the workflow:

1. The assistant classifies incoming emails
2. Certain decisions (NOTIFY classifications and response drafts) are routed to Agent Inbox
3. The workflow pauses until you provide input
4. Your decision determines the next step in the workflow
5. Your feedback is captured for potential improvement

This human-in-the-loop approach gives you oversight while still letting the assistant handle routine tasks, creating an effective collaboration.

## Memory & Learning Through Feedback

Our email assistant becomes even more powerful when we add memory capabilities, allowing it to learn from user feedback and adapt to preferences over time.

### Introducing Semantic Memory

The memory-enabled assistant (`email_assistant_hitl_memory.py`) uses [LangMem](https://langchain-ai.github.io/langmem/) to create four specialized memory types:

1. **Triage Preferences Memory**: Stores rules about how emails should be classified
2. **Response Preferences Memory**: Captures style and content preferences for email responses
3. **Calendar Preferences Memory**: Remembers scheduling preferences for meetings
4. **Background Memory**: Accumulates factual information about colleagues, projects, and context

These memory types work seamlessly with [LangGraph Store](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory), which is a simple database that is built into our local deployment that we get when we run `langGraph dev` locally or use the LangGraph Platform hosted service. The Store persists across each interaction (or thread) with the assistant, allowing the accumulation of memories over time. 

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

Over time, you can see memories accumulate in the `Memory` store viewing in LangGraph Studio.

![Screenshot 2025-04-04 at 4 08 03 PM](https://github.com/user-attachments/assets/2f8adbc5-9719-46df-a77d-d52c71c015dc)

## Deployment 

We've buit up to a system that can learn our preferences over time:

![iterrupt_conf_assistant](https://github.com/user-attachments/assets/b7ebe213-f214-4bdb-bb21-7c41edf01206)

The graph already can be run with LangGraph Platform locally using the `langgraph dev` command. 

## Customization 

You can use any model support by `init_chat_model` shown [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).
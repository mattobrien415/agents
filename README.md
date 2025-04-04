# Interrupt Workshop 

AI assistants promise to make work easier. But we need effective ways to teach them our preferences and benchmark their performance. In this workshop, we'll build a self-improving email assistant that can intelligently manage your e-mail inbox. It brings together a few components of the LangGraph ecosystem: LangGraph for agent orchestration, Agent Inbox for human-in-the-loop, LangMem for memory, and LangSmith for evaluation.

< add map >

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

We want a few core functions in a baseline e-mail assistant:

1. **Email Triage**: The assistant analyzes incoming emails and classifies them into three categories:
   - **Respond**: Emails that require a direct response
   - **Notify**: Important informational emails that should be noted
   - **Ignore**: Low-priority emails like marketing or spam that can be safely ignored

2. **Email Response**: For emails classified as needing a response, the assistant can craft and send appropriate replies using the available tools.

### Tool calling agent vs Workflow 

There are [a few approaches to building an email assistant](https://langchain-ai.github.io/langgraph/tutorials/workflows/), including:

1. **Tool-calling ReAct Agent** (`email_assistant_react.py`):
   - A single agent handles *all* tasks - both triage and response
   - Has a dedicated triage tool to categorize emails
   - Has tools for to e-mail drafting, calendar scheduling, and calendar search
   - All decision-making happens within one LLM component

2. **Agentic Workflow** (`email_assistant.py`):
   - Includes a clear separation of concerns between triage and response logic
   - Has router node to decide what to do with an email
   - E-mail response is handled by a separate agent with tools for e-mail drafting, calendar scheduling, and calendar search

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
      "email_assistant_react": "./src/email_assistant/email_assistant_react.py:email_assistant"
    },
```

![Screenshot 2025-04-01 at 3 38 24 PM](https://github.com/user-attachments/assets/f93aa02e-5497-440e-9040-eb149701226b)

In studio, you can test both assistants with some email inputs directly to see what the assistants will do:
```shell
{"author": "Alice Smith <alice.smith@company.com>",
  "to": "John Doe <john.doe@company.com>",
  "subject": "Quick question about API documentation",
  "email_thread": "Hi John, I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs? Specifically, I'm looking at: /auth/refres /auth/validate Thanks! Alice"}
```

```shell
{
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": "Dear Valued Developer,\n\nDon't miss out on this INCREDIBLE opportunity! \n\n🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! \n\n✨ FEATURES:\n- Revolutionary AI-powered code completion\n- Cloud-based development environment\n- 24/7 customer support\n- And much more!\n\n💰 Regular Price: $999/month\n🎉 YOUR SPECIAL PRICE: Just $199/month!\n\n🕒 Hurry! This offer expires in:\n24 HOURS ONLY!\n\nClick here to claim your discount: https://amazingdeals.com/special-offer\n\nBest regards,\nMarketing Team\n---\nTo unsubscribe, click here"
}
```

```shell
{
    "author": "System Admin <sysadmin@company.com>",
    "to": "Development Team <dev@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": "Hi team,\n\nThis is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.\n\nPlease plan your work accordingly and ensure no critical deployments are scheduled during this window.\n\nThanks,\nSystem Admin Team"
}
```

## Evaluation

### Structure 

The evaluation framework in the `eval` folder compares the performance of both assistant implementations on two key aspects:

1. **Dataset**: A collection of sample emails with ground truth classifications and responses is defined in `email_dataset.py`

2. **Email Triage Evaluation** (`evaluate_triage.py`):
   - Uses LangSmith to run and track evaluations
   - Creates a dataset of test emails if it doesn't exist
   - Runs both assistant implementations against the dataset
   - Uses direct string matching to evaluate if classifications match expected values
   - Scores assistants on a 0-1 scale based on exact matches

3. **Email Response Evaluation** (`evaluate_response.py`):
   - Uses the same LangSmith evaluation framework
   - Creates a response quality dataset if it doesn't exist
   - Uses an LLM-as-judge approach with the `RESPONSE_QUALITY_PROMPT` 
   - Assesses how well each assistant crafts appropriate responses
   - Scores response quality on a 0-1 scale

4. **Visualization**:
   - Each evaluation generates a comparative bar chart showing performance of both approaches
   - Saves results to `eval/results/` with timestamps
   - Provides clear metrics on which assistant performed better

### Run Evaluation 

To evaluate the performance of both email assistant implementations, follow these steps:

#### Prerequisites
1. Set up your LangSmith API key as an environment variable:
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key
   ```

#### Running the Triage Evaluation
Execute the triage evaluation script:
```bash
python -m eval.evaluate_triage
```

This script will:
1. Create a dataset in LangSmith with test emails defined in `eval/email_dataset.py` (if it doesn't exist)
2. Run both the agentic workflow and ReAct email assistants against this dataset
3. Evaluate each assistant's performance using an LLM-as-judge approach
4. Generate a visual comparison of the results and save it to `eval/results/`
5. Print the performance scores of both assistants in the terminal

#### Running the Response Quality Evaluation
Execute the response quality evaluation script:
```bash
python -m eval.evaluate_response
```

This script follows the same approach but focuses on evaluating the quality and appropriateness of email responses.

#### Understanding the Results
The triage evaluation measures how well each assistant correctly classifies emails as "respond", "notify", or "ignore" using direct string matching with the expected classifications. The response evaluation uses an LLM-as-judge approach to measure the quality and appropriateness of the responses generated. Both use a 0-1 scale, where higher scores indicate better performance.

#### Viewing Results in LangSmith
After running the evaluations, you can view detailed results in the LangSmith dashboard:
1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to the "Datasets" section to find the relevant datasets
3. View the results of each experiment to see detailed performance breakdowns

You can also find an example dataset with previous evaluation results [here](https://smith.langchain.com/public/1e3765c9-3455-4243-bb75-e4d865cc5960/d).

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

![Triage review interface](https://github.com/user-attachments/assets/160c357e-f4a8-4626-b74c-fef17b85127b)

#### 4. Response Review Options

When the assistant drafts an email response (for RESPOND classifications), you can:

* **Edit**: Modify the content of the response directly
* **Accept**: Send the response as drafted
* **Provide Feedback**: Offer guidance on how to improve similar responses
* **Dismiss**: Reject the response entirely

< Image >

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

## Deployment 

The graph already can be run with LangGraph Platform locally using the `langgraph dev` command. 

TODO: Add remote deployment instructions. 

## Customization 

You can use any model support by `init_chat_model` shown [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

TODO: Add to configuration 

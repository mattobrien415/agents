# Interrupt Workshop 

In this workshop, we will build an email assistant that can triage incoming emails and respond to them appropriately. The assistant will analyze email content to determine whether each message should be responded to, simply noted, or safely ignored, helping users manage their inbox more efficiently. This simple application will showcase many aspects of LangGraph, LangChain, and the LangSmith ecosystem, including the ability to build agents, evaluate them, add human in the loop, add memory, and deploy them using LangGraph Platform.

## Environment Setup 

### Prerequisites

1. Set up your LangSmith API key as an environment variable:
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key
   ```
2. Set up your OpenAI API key (default LLM used for the Evaluator):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```
3. Set up your Anthropic API key (default LLM used for the E-mail Assistant):
   ```bash
   export ANTHROPIC_API_KEY=your_anthropic_api_key
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

There are two main approaches to building an email assistant:

1. **Tool-calling ReAct Agent** (`email_assistant_react.py`):
   - A single agent handles all tasks - both triage and response
   - Uses a ReAct (Reasoning and Acting) pattern to decide what to do next
   - More flexible but potentially less structured
   - Has a dedicated triage tool to categorize emails
   - All decision-making happens within one LLM component

2. **Workflow** (`email_assistant.py`):
   - Uses a structured graph with specialized nodes for different tasks
   - Clear separation between triage and response logic
   - More explicit control flow with defined transitions
   - Potentially more maintainable and easier to debug
   - Can use different prompt templates and models for different steps

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
      "email_assistant_react": "./src/email_assistant/email_assistant_react.py:email_assistant_react"
    },
```

![Screenshot 2025-04-01 at 3 38 24 PM](https://github.com/user-attachments/assets/f93aa02e-5497-440e-9040-eb149701226b)

In studio, you can test some email inputs directly to see what the assistant will do:
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
   - Uses an LLM-as-judge approach with the `TRIAGE_CLASSIFICATION_PROMPT` from `prompt.py`
   - Evaluates whether each assistant correctly classified emails
   - Scores assistants on a 0-1 scale

3. **Email Response Evaluation** (`evaluate_response.py`):
   - Uses the same LangSmith evaluation framework
   - Creates a response quality dataset if it doesn't exist
   - Uses the `RESPONSE_QUALITY_PROMPT` to evaluate response quality
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

2. Set up your OpenAI API key (required for the evaluator):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

#### Running the Triage Evaluation
Execute the triage evaluation script:
```bash
python -m eval.evaluate_triage
```

This script will:
1. Create a dataset in LangSmith with test emails defined in `eval/email_dataset.py` (if it doesn't exist)
2. Run both the workflow-based and ReAct-based email assistants against this dataset
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
The triage evaluation measures how well each assistant correctly classifies emails as "respond", "notify", or "ignore". The response evaluation measures the quality and appropriateness of the responses generated. Both use a 0-1 scale, where higher scores indicate better performance.

#### Viewing Results in LangSmith
After running the evaluations, you can view detailed results in the LangSmith dashboard:
1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to the "Datasets" section to find the relevant datasets
3. View the results of each experiment to see detailed performance breakdowns

You can also find an example dataset with previous evaluation results [here](https://smith.langchain.com/public/1e3765c9-3455-4243-bb75-e4d865cc5960/d).

![Screenshot 2025-04-01 at 3 04 05 PM](https://github.com/user-attachments/assets/0545212b-4563-4ca8-a748-abe31c84ee18)

## Human-in-the-loop 

### Adding HITL to the Workflow 

Modify the `langgraph.json` file to point to the new graph that includes HITL: 
```shell
  "graphs": {
      "email_assistant": "./src/email_assistant/email_assistant.py:email_assistant_hitl",
    },
```

This allows us to interrupt the workflow after the triage decision and review it. For this, we use [Agent Inbox](https://github.com/langchain-ai/agent-inbox). First, start the server:
```shell
$ langgraph dev 
```

Then, open: 
```
https://dev.agentinbox.ai/
```

Select `add inbox` provide it:
* Graph name: the name from the `langgraph.json` file (`email_assistant`)
* Graph URL: `http://127.0.0.1:2024/`

Pass any of the the above inputs to your assistant in Studio, and you will see the thread in Agent Inbox. You can review the triage decision and respond to the email. 

### Agent Inbox Details

The HITL implementation works by pausing the workflow at the triage step to get user input on the decision. Here's how it works:

#### 1. Creating the Request

```python
request = {
    "action_request": {
        "action": "Review Triage",
        "args": {}
    },
    "config": {
        "allow_ignore": True,  
        "allow_respond": True, 
        "allow_edit": False, 
        "allow_accept": True, 
    },
    "description": email_markdown,
}
```

This creates a package of information that will be displayed in Agent Inbox:
- `action_request`: Describes what type of action this is ("Review Triage") and any initial values
- `config`: Defines what buttons will appear for the user in Agent Inbox:
  - `allow_ignore`: Shows a "Dismiss" or "Ignore" button
  - `allow_respond`: Shows a text input field for free-form responses
  - `allow_edit`: Shows an edit interface (disabled for triage review)
  - `allow_accept`: Shows an "Accept" button to approve the triage decision
- `description`: Contains the formatted email content and triage decision that will be displayed

#### 2. Sending to Agent Inbox

```python
response = interrupt([request])[0]
```

This line does several important things:
- It pauses the current execution of the workflow
- It sends the request to Agent Inbox
- The request appears as a new item in the user's Agent Inbox interface
- The workflow waits until the user interacts with the item
- When the user takes an action (ignores, responds, or accepts), that response is returned

#### 3. What the User Sees

The agent inbox has each thread from Studio:

![Screenshot 2025-04-02 at 4 37 06 PM](https://github.com/user-attachments/assets/e45e063b-6e54-49b7-8fef-a0280b52e683)

The user will see a new item in their Agent Inbox with:
- The email content formatted nicely (from email_markdown)
- The triage decision (e.g., "📧 Classification: RESPOND - This email requires a response")
- A set of action buttons based on the config settings
- A text input field if allow_respond is True

![Screenshot 2025-04-02 at 4 14 31 PM](https://github.com/user-attachments/assets/160c357e-f4a8-4626-b74c-fef17b85127b)

#### 4. Handling User Responses

The workflow handles different user responses:

```python
# Accept the decision to respond  
if response["type"] == "accept":
    # Go to the response agent
    goto = "response_agent"
    # Add the email to the messages
    messages.append({"role": "user",
                     "content": f"Respond to the email {state['email_input']}"
                     })
# Ignore the email 
elif response["type"] == "ignore":
    goto = END
    # Add the email to the messages
    messages.append({"role": "user",
                     "content": "User feedback: Ignore email"
                     })
elif response["type"] == "response":
    # Add user_input to memory
    user_input = response["args"]
    messages.append({"role": "user",
                     "content": f"User feedback: {user_input}"
                     })
    goto = END
```

This mechanism creates a clean user experience where:
1. The assistant classifies an email (respond, ignore, or notify)
2. The user reviews this decision in Agent Inbox with appropriate options
3. The workflow waits for their input
4. Once they respond, the workflow continues executing with their response
5. The user's feedback is captured in the messages for potential future use

In our implementation, we've customized the UI options for different triage decisions. For example, when an email is classified as "respond", we don't show the "ignore" button, encouraging the user to either accept the decision or provide feedback.

## Memory 

TODO: Add memory to the workflow. 

## Deployment 

The graph already can be run with LangGraph Platform locally using the `langgraph dev` command. 

TODO: Add remote deployment instructions. 

## Customization 

You can use any model support by `init_chat_model` shown [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

Simply modify the `llm` variable in the respective files:
* `src/email_assistant/email_assistant_react.py`
* `src/email_assistant/email_assistant.py`
* `eval/evaluate_triage.py`

You can also modify the assistant and evaluator prompts in:
* `src/email_assistant/prompts.py` 
* `src/eval/prompt.py`

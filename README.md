# Interrupt Workshop 

In this workshop, we will build an email assistant that can triage incoming emails and respond to them appropriately. The assistant will analyze email content to determine whether each message should be responded to, simply noted, or safely ignored, helping users manage their inbox more efficiently. This simple application will showcase many aspects of LangGraph, LangChain, and the LangSmith ecosystem, including the ability to build agents, evaluate them, add human in the loop, add memory, and deploy them using LangGraph Platform.

## Environment Setup 

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

![Screenshot 2025-04-01 at 3 38 24 PM](https://github.com/user-attachments/assets/f93aa02e-5497-440e-9040-eb149701226b)

Here we can see the two assistants easily and test them.

Pass in an example email to either assistant to see the output:
```shell
{
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": """Dear Valued Developer,

Don't miss out on this INCREDIBLE opportunity! 

🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

✨ FEATURES:
- Revolutionary AI-powered code completion
- Cloud-based development environment
- 24/7 customer support
- And much more!

💰 Regular Price: $999/month
🎉 YOUR SPECIAL PRICE: Just $199/month!

🕒 Hurry! This offer expires in:
24 HOURS ONLY!

Click here to claim your discount: https://amazingdeals.com/special-offer

Best regards,
Marketing Team
---
To unsubscribe, click here
""",
}
```

## Evaluation

### Structure 

The evaluation framework in the `eval` folder compares the performance of both assistant implementations on the email triage task:

1. **Dataset**: A collection of sample emails with ground truth classifications is defined in `email_dataset.py`

2. **Evaluation Process** (`evaluate_triage.py`):
   - Uses LangSmith to run and track evaluations
   - Creates a dataset of test emails if it doesn't exist
   - Runs both assistant implementations against the dataset
   - Uses an LLM-as-judge approach with a specialized prompt from `prompt.py`
   - Evaluates whether each assistant correctly classified emails
   - Scores assistants on a 0-1 scale

3. **Visualization**:
   - Generates a comparative bar chart showing performance of both approaches
   - Saves results to `eval/results/` with timestamps
   - Provides clear metrics on which assistant performed better

### Run 

```
python -m eval.evaluate_triage
```

This will kick off evaluation on a dataset defined in eval/triage_dataset.py. We will load ths into LangSmith and then evaluate both agents on it.

We can compare the results in LangSmith. [Here](https://smith.langchain.com/public/1e3765c9-3455-4243-bb75-e4d865cc5960/d) is an example dataset link.

![Screenshot 2025-04-01 at 3 04 05 PM](https://github.com/user-attachments/assets/0545212b-4563-4ca8-a748-abe31c84ee18)

## Human-in-the-loop 

## Memory 

## Deployment 

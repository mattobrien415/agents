# Agents From Scratch 

AI agents promise to transform how we work, but there's often a gap between hype and reality: to act on our behalf, agents need to learn and remember our preferences. The repo is a guide to building agents from scratch, building from simple principles to self-improving and personalized agents that use LangChain, LangGraph, and LangSmith. 

We're going to build an agent that can act an an e-mail assistant, because this is often a tedious task that could benefit from an AI assistant, but it requires a high level of personalization (e.g., what to respond to, what to ignore, what to schedule a meeting for, and how to respond). The ideas and approaches shown here can be applied to other agents across a wide range of tasks. Here is a map of the components covered:

![overview](notebooks/img/overview.png)

## Environment Setup 

### Python Version

* Ensure you're using Python 3.11 or later. 
* This version is required for optimal compatibility with LangGraph. 

```shell
python3 --version
```

### API Keys

* If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
* Sign up for LangSmith [here](https://smith.langchain.com/).
* Generate a LangSmith API key.

### Set Environment Variables

* Create a `.env` file in the root directory:
```shell
# Copy the .env.example file to .env
cp .env.example .env
```

* Edit the `.env` file with the following:
```shell
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="interrupt-workshop"
OPENAI_API_KEY=your_openai_api_key
```

* You can also set the environment variables in your terminal:
```shell
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_TRACING=true
export OPENAI_API_KEY=your_openai_api_key
```

### Create a virtual environment and activate it

```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
# Ensure you have a recent version of pip (required for editable installs with pyproject.toml)
$ python3 -m pip install --upgrade pip
# Install the package in editable mode
$ pip install -e .
```

## Structure 

The repo is organized into the 4 sections, with a notebook for each and accompanying code in the `src/email_assistant` directory.

### Building an agent 
* Notebook: `notebooks/agent.ipynb`
* `src/email_assistant/email_assistant.py`

![overview-agent](notebooks/img/overview_agent.png)

In this section, we review the philosophy of building agents, thinking about which parts we can encode as a [fixed workflow](https://langchain-ai.github.io/langgraph/tutorials/workflows/) and which need to be an agent. We compare a tool-calling agent to an agentic workflow, which has a dedicated router to handle the email triage step and allows the agent to focus on the email response. We introduce LangGraph Platform, which can be used to run both of them locally:
```shell
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

![Screenshot 2025-04-04 at 4 06 18 PM](notebooks/img/studio.png)

### Evaluation 
* Notebook: `notebooks/evaluation.ipynb`
* `eval` and `tests` directories

![overview-eval](notebooks/img/overview_eval.png)

We introduce a collection of sample emails with ground truth classifications, responses, and expected tool calls defined in `eval/email_dataset.py`. We then use this dataset to test the two assistants above using both Pytest and LangSmith `evaluate` API. The `tests/run_all_tests.py` script can be used to run Pytest on all examples for each assistant in this repo.

```bash
# Run with rich output display
python -m tests.run_all_tests --rich-output
```

![Screenshot 2025-04-08 at 8 07 48 PM](notebooks/img/eval.png)

In additon, the `evaluate_triage.py` script will run the triage evaluation using the LangSmith `evaluate` API and log the results to LangSmith:

```bash
python -m eval.evaluate_triage
```

### Human-in-the-loop 
* Notebook: `notebooks/hitl.ipynb`
* `src/email_assistant/email_assistant_hitl.py`

![overview-hitl](notebooks/img/overview_hitl.png)

What if we want the ability to review and correct the assistant's decisions? In this section, we show how to add a human-in-the-loop (HITL) to the assistant. For this, we use [Agent Inbox](https://github.com/langchain-ai/agent-inbox) to review and correct the assistant's decisions.

![Agent Inbox showing email threads](notebooks/img/agent-inbox.png)


### Memory & Learning Through Feedback 
* Notebook: `notebooks/memory.ipynb`
* `src/email_assistant/email_assistant_hitl_memory.py`

![overview-memory](notebooks/img/overview_memory.png)  

Our email assistant becomes more powerful when we add memory capabilities, allowing it to learn from user feedback and adapt to preferences over time. The memory-enabled assistant (`email_assistant_hitl_memory.py`) uses [LangMem](https://langchain-ai.github.io/langmem/) to manage memories seamlessly with [LangGraph Store](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory). Over time, you can see memories accumulate in the `Memory` store when viewing in LangGraph Studio.

### Deployment 

We've built up to a system that can learn our preferences over time. The graph can be run locally and deployed to LangGraph Platform for production use.

#### Local Development

Run the application locally using LangGraph Platform:

```shell
# Install langgraph CLI
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Deploying to LangGraph Platform

1. Navigate to the deployments page in LangSmith
2. Click "New Deployment"
3. Connect to your GitHub repository containing this code
4. Give your deployment a name (e.g., "Email-Assistant")
5. Add the necessary environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GMAIL_TOKEN`: JSON content from your Gmail token file (for Gmail integration)
   - `GMAIL_SECRET`: JSON content from your Gmail secrets file (for Gmail integration)
6. Click "Submit"
7. Once deployed, you'll receive a URL for your deployment

#### Setting Up Gmail Integration

For production use with real emails, follow these steps:

1. Set up Google API credentials following the instructions in [Gmail Tools README](src/email_assistant/tools/gmail/README.md)
2. Test your Gmail integration locally:
   ```shell
   python src/email_assistant/tools/gmail/run_ingest.py --email your@email.com --minutes-since 1440 --include-read
   ```
3. Once deployed, connect to your LangGraph deployment:
   ```shell
   python src/email_assistant/tools/gmail/run_ingest.py --email your@email.com --minutes-since 1440 --include-read --url https://your-deployment-url.us.langgraph.app
   ```

#### Setting Up Automated Email Processing

To automatically process emails on a schedule:

1. Configure a cron job using the provided setup script:
   ```shell
   python src/email_assistant/tools/gmail/setup_cron.py \
     --email your@email.com \
     --url https://your-deployment-url.us.langgraph.app \
     --minutes-since 60 \
     --schedule "0 * * * *" \
     --include-read
   ```
   This will set up an hourly job that processes emails from the past hour.

2. Monitor your automated jobs through the LangGraph Studio UI.

Full documentation for Gmail integration and deployment is available in the [Gmail Tools README](src/email_assistant/tools/gmail/README.md).

## Tools and Integrations

The email assistant uses a modular tools architecture that allows for different implementations and integrations to be easily swapped. This is structured in the `src/email_assistant/tools` directory:

- **Default Tools**: By default, the assistant uses mock email and calendar tools for testing and development. These are located in `src/email_assistant/tools/default/`.

- **Gmail Integration**: For connecting to real email and calendar services, the assistant uses Gmail API integration tools in `src/email_assistant/tools/gmail/`. This integration allows the assistant to:
  - Fetch real emails from Gmail
  - Send replies to email threads
  - Check calendar availability and schedule meetings
  - Automate email processing with cron jobs
  
  See the [Gmail Tools README](src/email_assistant/tools/gmail/README.md) for detailed setup instructions and the Deployment section above for production configuration.

- **Custom Integrations**: The modular architecture makes it easy to add new tool integrations by following the same pattern as the existing tool packages.

To use a specific set of tools in your agent:

```python
# For default tools only
tools = get_tools()

# For including Gmail tools
tools = get_tools(include_gmail=True)

# For a specific subset of tools
tools = get_tools(tool_names=["write_email", "triage_email", "fetch_emails_tool"])
```

For a complete example of using Gmail tools, see `src/email_assistant/gmail_assistant.py`.

### Agent Inbox

When using human-in-the-loop capabilities, you can view and interact with the agent at:
https://dev.agentinbox.ai/

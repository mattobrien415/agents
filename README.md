# Interrupt Workshop 

AI agents promise to transform how we work, but there's often a gap between hype and reality: to act on our behalf, agents need to learn and remember our preferences. In this workshop, we're going to show how to build self-improving and personalized agents from scratch using LangChain, LangGraph, and LangSmith. 

We're going to build an agent that can act an an e-mail assistant, because this is often a tedious task that could benefit from an AI assistant, but it requires a high level of personalization (e.g., what to respond to, what to ignore, what to schedule a meeting for, and how to respond). The ideas and approaches shown here can be applied to other agents across a wide range of tasks. Here is a map of the components covered in the workshop:

![interrupt_conf_high_level](notebooks/img/overview.png)

## Environment Setup 

### Prerequisites

1. Set up your LangSmith API key as an environment variable:
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key
   ```
2. Set up your OpenAI API key (default; you can also use other LLMs):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

Create a virtual environment and activate it:
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

## Structure 

The workshop is organized into the 4 sections, with a notebook for each and accompanying code in the `src/email_assistant` directory.

### Building an agent 
* Notebook: `notebooks/agent.ipynb`
* `src/email_assistant/baseline_agent.py`
* `src/email_assistant/email_assistant.py`

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

We introduce a collection of sample emails with ground truth classifications, responses, and expected tool calls defined in `eval/email_dataset.py`. We then use this dataset to test the two assistants above using both Pytest and LangSmith `evaluate` API. The `run_all_tests.py` script can be used to run Pytest on all examples for each assistant in this repo.

```bash
# Run with rich output display
python run_all_tests.py --rich-output
```

![Screenshot 2025-04-08 at 8 07 48 PM](notebooks/img/eval.png)

In additon, the `evaluate_triage.py` script will run the triage evaluation using the LangSmith `evaluate` API and log the results to LangSmith:

```bash
python -m eval.evaluate_triage
```

### Human-in-the-loop 
* Notebook: `notebooks/hitl.ipynb`
* `src/email_assistant/email_assistant_hitl.py`

What if we want the ability to review and correct the assistant's decisions? In this section, we show how to add a human-in-the-loop (HITL) to the assistant. For this, we use [Agent Inbox](https://github.com/langchain-ai/agent-inbox) to review and correct the assistant's decisions.

![Agent Inbox showing email threads](notebooks/img/agent-inbox.png)


### Memory & Learning Through Feedback 
* Notebook: `notebooks/memory.ipynb`
* `src/email_assistant/email_assistant_hitl_memory.py`

Our email assistant becomes more powerful when we add memory capabilities, allowing it to learn from user feedback and adapt to preferences over time. The memory-enabled assistant (`email_assistant_hitl_memory.py`) uses [LangMem](https://langchain-ai.github.io/langmem/) to manage memories seamlessly with [LangGraph Store](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory). Over time, you can see memories accumulate in the `Memory` store when viewing in LangGraph Studio.

### Deployment 

We've built up to a system that can learn our preferences over time. The graph already can be run with LangGraph Platform locally using the `langgraph dev` command and deployed to the LangGraph Platform hosted service.
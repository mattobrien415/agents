from typing import Literal

from langchain.chat_models import init_chat_model

from src.email_assistant.tools import get_tools, get_tools_by_name
from src.email_assistant.tools.default.prompt_templates import STANDARD_TOOLS_PROMPT
from src.email_assistant.prompts import agent_system_prompt_baseline, default_background, default_response_preferences, default_cal_preferences, default_triage_instructions
from src.email_assistant.schemas import State

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Get tools
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# Initialize the LLM, enforcing tool use (of any available tools)
llm = init_chat_model("openai:gpt-4o", tool_choice="required", temperature=0.0)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# LLM call node
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_baseline.format(
                        tools_prompt=STANDARD_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences, 
                        triage_instructions=default_triage_instructions
                    )}
                ]
                + state["messages"]
            )
        ]
    }

# Tool handling node
def tool_handler(state: State):
    """Process tool calls and execute them"""
    
    # Store messages
    result = []
    # Track if we need to update classification_decision
    classification_decision = None
    
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Execute the tool
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        # TODO: These are still hard-coded and would need to be updated if we swap out tools used 
        # Log the classification decision if triage_email was called
        if tool_call["name"] == "triage_email":
            classification_decision = tool_call["args"]["category"]
            
        # Add the tool response
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    
    # Return updated state with both messages and classification_decision if available
    if classification_decision:
        return {
            "messages": result,
            "classification_decision": classification_decision
        }
    else:
        return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["tool_handler", END]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "tool_handler"

# Build workflow
overall_workflow = StateGraph(State,input=MessagesState)

# Add nodes
overall_workflow.add_node("llm_call", llm_call)
overall_workflow.add_node("tool_handler", tool_handler)

# Add edges
overall_workflow.add_edge(START, "llm_call")
overall_workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_handler": "tool_handler",
        END: END,
    },
)
overall_workflow.add_edge("tool_handler", "llm_call")

# Compile the agent
email_assistant = overall_workflow.compile()
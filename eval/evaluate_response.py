from langsmith import Client
from langsmith import testing as t

import os
import matplotlib.pyplot as plt
from datetime import datetime

from openevals.llm import create_llm_as_judge
from eval.prompt import RESPONSE_QUALITY_PROMPT
from eval.email_dataset import examples_response

from email_assistant.email_assistant import email_assistant as email_assistant_workflow 
from email_assistant.email_assistant_react import email_assistant as email_assistant_react
from email_assistant.utils import format_messages

# Client 
client = Client()

# Dataset name
dataset_name = "Interrupt Workshop: E-mail Response Dataset"

# If the dataset doesn't exist, create it
if not client.has_dataset(dataset_name=dataset_name):

    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="A dataset of e-mails and their response quality evaluations."
    )

    # Add examples to the dataset
    client.create_examples(dataset_id=dataset.id, examples=examples_response)

# Target functions that run our email assistants
def target_email_assistant(inputs: dict) -> dict:
    """Process an email through the workflow-based email assistant.
    
    Args:
        inputs: A dictionary containing the email_input field from the dataset
        
    Returns:
        A formatted dictionary with the assistant's response messages
    """
    response = email_assistant_workflow.invoke({"email_input": inputs["email_input"]})
    return format_messages(response['messages'])

def target_email_assistant_react(inputs: dict) -> dict:
    """Process an email through the ReAct-based email assistant.
    
    Args:
        inputs: A dictionary containing the email_input field from the dataset
        
    Returns:
        A formatted dictionary with the assistant's response messages
    """
    messages = [{"role": "user", "content": str(inputs["email_input"])}]
    response = email_assistant_react.invoke({"messages": messages})
    return format_messages(response['messages'])

## Evaluator 
feedback_key = "response_quality" # Key saved to langsmith

def response_quality_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate the assistant's email response quality against reference outputs.
    
    This evaluator uses LLM-as-judge to compare the model's response 
    with the reference response. It evaluates:
    - Whether the assistant correctly handled the response
    - The quality and appropriateness of the assistant's response
    
    Args:
        inputs: The original email input from the dataset
        outputs: The assistant's response from the target function
        reference_outputs: The ground truth response from the dataset
        
    Returns:
        An evaluation result with feedback on the assistant's performance
    """

    evaluator = create_llm_as_judge(
        prompt=RESPONSE_QUALITY_PROMPT,
        feedback_key=feedback_key, 
        continuous=True, # Set 0-1 scale by default
        model="openai:o3-mini",
    )
    return evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

## Run evaluation
experiment_results_react = client.evaluate(
    # Run agent  
    target_email_assistant_react,
    # Dataset name  
    data=dataset_name,
    # Evaluator
    evaluators=[
        response_quality_evaluator
    ],
    # Name of the experiment
    experiment_prefix="E-mail assistant react", 
    # Number of concurrent evaluations
    max_concurrency=2, 
)

experiment_results_workflow = client.evaluate(
    # Run agent 
    target_email_assistant,
    # Dataset name   
    data=dataset_name,
    # Evaluator
    evaluators=[
        response_quality_evaluator
    ],
    # Name of the experiment
    experiment_prefix="E-mail assistant workflow", 
    # Number of concurrent evaluations
    max_concurrency=2, 
)

## Add visualization
# Convert evaluation results to pandas dataframes
df_react = experiment_results_react.to_pandas()
df_workflow = experiment_results_workflow.to_pandas()

# Calculate mean scores (values are on a 0-1 scale)
react_score = df_react[f'feedback.{feedback_key}'].mean()
workflow_score = df_workflow[f'feedback.{feedback_key}'].mean()

# Create a bar plot comparing the two models
plt.figure(figsize=(10, 6))
models = ['ReAct Agent', 'Workflow Agent']
scores = [react_score, workflow_score]

# Create bars with distinct colors
plt.bar(models, scores, color=['#5DA5DA', '#FAA43A'], width=0.5)

# Add labels and title
plt.xlabel('Agent Type')
plt.ylabel('Average Score')
plt.title(f'Email Response Quality Comparison')

# Add score values on top of bars
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f'{score:.2f}', ha='center', fontweight='bold')

# Set y-axis limit
plt.ylim(0, 1.1)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure the output directory exists
os.makedirs('eval/results', exist_ok=True)

# Save the plot with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f'eval/results/response_comparison_{timestamp}.png'
plt.savefig(plot_path)
plt.close()

print(f"\nEvaluation visualization saved to: {plot_path}")
print(f"ReAct Agent Score: {react_score:.2f}")
print(f"Workflow Agent Score: {workflow_score:.2f}")
# Used in /eval/evaluate_triage.py
TRIAGE_CLASSIFICATION_PROMPT = """

<Task>
You are evaluating the classification of emails.

They should be be classified into one of the following categories:
- ignore
- notify
- respond

You will be given:
- the email_input
- the agent's reasoning and decision as a list of messages 
- the correct classification

Your job is to evaluate the agent's reasoning and decision relative to the correct classification.
</Task>

<email_input>
{inputs}
</email_input>

<agent_response>
{outputs}
</agent_response>

<correct_classification>
{reference_outputs}
</correct_classification>
"""

# Used in /tests/test_email_assistant.py
RESPONSE_CRITERIA_SYSTEM_PROMPT = """You are evaluating an email assistant's response to determine if it meets specific criteria.
                 
The assistant should respond appropriately to emails based on their content. Your job is to evaluate if the assistant's final response meets the criteria provided.

Be objective and fair, providing specific examples from the response to justify your evaluation."""

# Used in /tests/test_email_assistant.py
HITL_FEEDBACK_SYSTEM_PROMPT = """You are evaluating an email assistant's response to determine if it meets specific criteria.

This is an email assistant that is used to respond to emails. Review our initial email response and the user feedback given to update the email response. Here is the feedback: {feedback}. Assess whether the final email response addresses the feedback that we gave."""

# Used in /tests/test_email_assistant.py
MEMORY_UPDATE_SYSTEM_PROMPT = """This is an email assistant that uses memory to update its response preferences. 

Review the initial response preferences and the updated response preferences. Assess whether the updated response preferences are more accurate than the initial response preferences."""
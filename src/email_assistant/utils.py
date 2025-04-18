from typing import List, Any
import io
import sys
import json

def format_email_markdown(subject, author, to, email_thread):
    """Format email details into a nicely formatted markdown string for display"""
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}

{email_thread}

---
"""

def format_for_display(state, tool_call):
    """Format content for display in Agent Inbox
    
    Args:
        state: Current message state
        tool_call: The tool call to format
    """
    # Initialize empty display
    display = ""
    
    # Add tool call information
    if tool_call["name"] == "write_email":
        display += f"""# Email Draft

**To**: {tool_call["args"].get("to")}
**Subject**: {tool_call["args"].get("subject")}

{tool_call["args"].get("content")}
"""
    elif tool_call["name"] == "schedule_meeting":
        display += f"""# Calendar Invite

**Meeting**: {tool_call["args"].get("subject")}
**Attendees**: {', '.join(tool_call["args"].get("attendees"))}
**Duration**: {tool_call["args"].get("duration_minutes")} minutes
**Day**: {tool_call["args"].get("preferred_day")}
"""
    elif tool_call["name"] == "Question":
        # Special formatting for questions to make them clear
        display += f"""# Question for User

{tool_call["args"].get("content")}
"""
    else:
        # Generic format for other tools
        display += f"""# Tool Call: {tool_call["name"]}

Arguments:"""
        
        # Check if args is a dictionary or string
        if isinstance(tool_call["args"], dict):
            display += f"\n{json.dumps(tool_call['args'], indent=2)}\n"
        else:
            display += f"\n{tool_call['args']}\n"
    return display

def parse_email(email_input: dict) -> tuple[str, str, str, str]:
    """Parse an email input dictionary, supporting multiple schemas.

    Supports both standard schema (author, to, subject, email_thread) and 
    Gmail-specific schema (from_email, to_email, subject, page_content).

    Args:
        email_input (dict): Dictionary containing email fields in either format:
            Standard schema:
                - author: Sender's name and email
                - to: Recipient's name and email
                - subject: Email subject line
                - email_thread: Full email content
            Gmail schema:
                - from_email: Sender's email
                - to_email: Recipient's email
                - subject: Email subject line
                - page_content: Full email content
                - id: Gmail message ID
                - thread_id: Gmail thread ID
                - send_time: Time the email was sent

    Returns:
        tuple[str, str, str, str]: Tuple containing:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content
    """
    # Detect schema based on keys present in the input
    if "author" in email_input and "email_thread" in email_input:
        # Standard schema
        return (
            email_input["author"],
            email_input["to"],
            email_input["subject"],
            email_input["email_thread"],
        )
    elif "from_email" in email_input and "page_content" in email_input:
        # Gmail schema
        return (
            email_input["from_email"],
            email_input["to_email"],
            email_input["subject"],
            email_input["page_content"],
        )
    else:
        # Unknown schema, try to handle gracefully by looking for equivalent fields
        author = email_input.get("author") or email_input.get("from_email") or "Unknown Sender"
        to = email_input.get("to") or email_input.get("to_email") or "Unknown Recipient"
        subject = email_input.get("subject") or "No Subject"
        content = (
            email_input.get("email_thread") or 
            email_input.get("page_content") or 
            email_input.get("content") or 
            "No content available"
        )
        return (author, to, subject, content)

def extract_message_content(message) -> str:
    """Extract content from different message types as clean string.
    
    Args:
        message: A message object (HumanMessage, AIMessage, ToolMessage)
        
    Returns:
        str: Extracted content as clean string
    """
    content = message.content
    
    # Check for recursion marker in string
    if isinstance(content, str) and '<Recursion on AIMessage with id=' in content:
        return "[Recursive content]"
    
    # Handle string content
    if isinstance(content, str):
        return content
        
    # Handle list content (AIMessage format)
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
        return "\n".join(text_parts)
    
    # Don't try to handle recursion to avoid infinite loops
    # Just return string representation instead
    return str(content)

def format_few_shot_examples(examples):
    """Format examples into a readable string representation.

    Args:
        examples (List[Item]): List of example items from the vector store, where each item
            contains a value string with the format:
            'Email: {...} Original routing: {...} Correct routing: {...}'

    Returns:
        str: A formatted string containing all examples, with each example formatted as:
            Example:
            Email: {email_details}
            Original Classification: {original_routing}
            Correct Classification: {correct_routing}
            ---
    """
    formatted = []
    for example in examples:
        # Parse the example value string into components
        email_part = example.value.split('Original routing:')[0].strip()
        original_routing = example.value.split('Original routing:')[1].split('Correct routing:')[0].strip()
        correct_routing = example.value.split('Correct routing:')[1].strip()
        
        # Format into clean string
        formatted_example = f"""Example:
Email: {email_part}
Original Classification: {original_routing}
Correct Classification: {correct_routing}
---"""
        formatted.append(formatted_example)
    
    return "\n".join(formatted)

def extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool call names from messages, safely handling messages without tool_calls."""
    tool_call_names = []
    for message in messages:
        # Check if message is a dict and has tool_calls
        if isinstance(message, dict) and message.get("tool_calls"):
            tool_call_names.extend([call["name"].lower() for call in message["tool_calls"]])
        # Check if message is an object with tool_calls attribute
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_names.extend([call["name"].lower() for call in message.tool_calls])
    
    return tool_call_names

def format_messages_string(messages: List[Any]) -> str:
    """Format messages into a single string for analysis."""
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Run the pretty_print calls
    for m in messages:
        m.pretty_print()
    
    # Get the captured output
    output = new_stdout.getvalue()
    
    # Restore original stdout
    sys.stdout = old_stdout
    
    return output
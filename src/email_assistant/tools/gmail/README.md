# Gmail Integration Tools

This directory contains tools for integrating with Gmail and Google Calendar APIs to enable the email assistant to work with real emails and calendar events.

## Features

- **Email Fetching**: Retrieve recent emails from your Gmail account
- **Email Sending**: Send replies to email threads
- **Calendar Availability**: Check your Google Calendar for availability on specific dates
- **Meeting Scheduling**: Create calendar events and send invites to attendees

## Setup Instructions

### 1. Set up Google Cloud Project and Enable Gmail API

1. Enable the Gmail API by clicking the blue "Enable API" button [here](https://developers.google.com/gmail/api/quickstart/python#enable_the_api)
2. Configure the OAuth consent screen:
   - If you're using a personal email (non-Google Workspace), select "External" as the User Type
   - Add your email as a test user under "OAuth consent screen" > "Test users" to avoid the "App has not completed verification" error
   - The "Internal" option only works for Google Workspace accounts

### 2. Create Credentials

1. In the Google Cloud Console, navigate to "Credentials"
2. Click "Create Credentials" and select "OAuth client ID"
3. Choose "Desktop application" as the application type
4. Name your OAuth client and click "Create"
5. Download the client secret JSON file

### 3. Set Up Authentication Files

```bash
# Create a secrets directory
mkdir -p src/email_assistant/tools/gmail/.secrets

# Move your downloaded client secret to the secrets directory
mv /path/to/downloaded/client_secret.json src/email_assistant/tools/gmail/.secrets/secrets.json

# Run the Gmail setup script
python src/email_assistant/tools/gmail/setup_gmail.py
```

The setup script will:
1. Open a browser window for you to authenticate with your Google account
2. Generate a `token.json` file in the `.secrets` directory
3. This token will be used for Gmail API access

### 4. Run the Gmail Ingestion Script

Once you have authentication set up, you can run the Gmail ingestion script to fetch emails and process them with your email assistant.

#### Local 

1. Run the graph locally:
```
langgraph dev
```

2. Run ingestion script:
```bash
# Set your email address as an environment variable (or use --email parameter)
export EMAIL_ADDRESS=your.email@gmail.com

# Basic usage (defaults to email_assistant_hitl_memory graph)
python src/email_assistant/tools/gmail/run_ingest.py

# Parameters 
python src/email_assistant/tools/gmail/run_ingest.py --minutes-since 60 --rerun 1 --early 0 --email your.email@gmail.com
```

#### Important Parameters:
- `--graph-name`: Name of the LangGraph to use (default: "email_assistant_hitl_memory")
- `--email`: The email address to fetch messages from (alternative to setting EMAIL_ADDRESS)
- `--minutes-since`: Only process emails that are newer than this many minutes (default: 60)
- `--url`: URL of the LangGraph deployment (default: http://127.0.0.1:2024)
- `--log-dir`: Directory to store email logs (default: "email_logs")

Note: If you encounter a "Token has been expired or revoked" error, delete the existing `token.json` file and run the setup script again to generate a fresh token.

## Using Gmail Tools in Your Agent

To use Gmail tools in your agent, modify your agent code as follows:

```python
from src.email_assistant.tools import get_tools, get_tools_by_name
from src.email_assistant.tools.gmail.prompt_templates import COMBINED_TOOLS_PROMPT

# Get tools with Gmail integration enabled
tools = get_tools(include_gmail=True)
tools_by_name = get_tools_by_name(tools)

# Use the combined tools prompt in your agent's system prompt
system_prompt = agent_system_prompt.format(
    tools_prompt=COMBINED_TOOLS_PROMPT,
    # other parameters...
)
```

See `src/email_assistant/gmail_assistant.py` for a complete example.
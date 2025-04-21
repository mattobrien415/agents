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

1. Start the LangGraph server in one terminal:
```
cd /path/to/project
langgraph start
```

2. Run ingestion script in another terminal:
```bash
# Set your email address as an environment variable (or use --email parameter)
export EMAIL_ADDRESS=your.email@gmail.com

# Basic usage (defaults to email_assistant_hitl_memory graph)
python src/email_assistant/tools/gmail/run_ingest.py

# Parameters 
python src/email_assistant/tools/gmail/run_ingest.py --minutes-since 1000 --rerun 1 --early 0 --email you.email@gmail.com
```

> **Note:** If you don't want to run the LangGraph server, you can use the `--mock` flag to test the email fetching functionality without processing emails through LangGraph.

#### Important Parameters:
- `--graph-name`: Name of the LangGraph to use (default: "email_assistant_hitl_memory")
- `--email`: The email address to fetch messages from (alternative to setting EMAIL_ADDRESS)
- `--minutes-since`: Only process emails that are newer than this many minutes (default: 60)
- `--url`: URL of the LangGraph deployment (default: http://127.0.0.1:2024)
- `--log-dir`: Directory to store email logs (default: "email_logs")
- `--rerun`: Process emails that have already been processed (1=yes, 0=no, default: 0)
- `--early`: Stop after processing one email (1=yes, 0=no, default: 0)
- `--mock`: Run in mock mode without requiring a LangGraph server
- `--include-read`: Include emails that have already been read (by default only unread emails are processed)
- `--skip-filters`: Process all emails without filtering (by default only latest messages in threads where you're not the sender are processed)

#### Flag Combinations:
- `--rerun 1 --early 0`: Process all emails, including ones previously processed by LangGraph
- `--rerun 0 --early 1`: Process only one new (previously unprocessed) email and stop
- `--rerun 1 --early 1`: Process one email (regardless if it was processed before) and stop
- `--rerun 0 --early 0`: Process only new (previously unprocessed) emails
- `--include-read --skip-filters`: Process all emails, including ones marked as read and ones that would normally be filtered out
- `--minutes-since 1000 --include-read --skip-filters`: Process all emails from the past ~16 hours without any filtering

#### Troubleshooting:

- **Missing emails?** The Gmail API applies filters to show only important/primary emails by default. You can:
  - Increase the `--minutes-since` parameter to a larger value (e.g., 1000) to fetch emails from a longer time period
  - Use the `--include-read` flag to process emails marked as "read" (by default only unread emails are processed)
  - Use the `--skip-filters` flag to include all messages (not just the latest in a thread, and including ones you sent)
  - Try running with all options to process everything: `--include-read --skip-filters --minutes-since 1000`
  - Use the `--mock` flag to test the system with simulated emails

- **Connection errors:** If you get "Connection refused" or "All connection attempts failed" errors:
  - Make sure the LangGraph server is running with `langgraph start` in a separate terminal
  - Verify the port number matches in your script (default is 2024)
  - Use the `--mock` flag to test without a LangGraph server: `--mock`

- **Authentication issues:** If you encounter a "Token has been expired or revoked" error, delete the existing `token.json` file and run the setup script again to generate a fresh token.

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
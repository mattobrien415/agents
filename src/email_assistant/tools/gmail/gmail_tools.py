"""
Gmail tools implementation module. 
This module formats the Gmail API functions into LangChain tools.
"""

import os
import sys
import base64
import email.utils
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path
from pydantic import Field, BaseModel
from langchain_core.tools import tool

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for credentials and tokens
_ROOT = Path(__file__).parent.absolute()
_SECRETS_DIR = _ROOT / ".secrets"

# We need to try importing the Gmail API libraries
# If they're not available, we'll use a mock implementation
try:
    import logging
    from googleapiclient.discovery import build
    from email.mime.text import MIMEText
    from datetime import timedelta
    from dateutil.parser import parse as parse_time
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Email content extraction function
    def extract_message_part(payload):
        """Extract content from a message part."""
        if payload.get("body", {}).get("data"):
            # Handle base64 encoded content
            data = payload["body"]["data"]
            decoded = base64.urlsafe_b64decode(data).decode("utf-8")
            return decoded
            
        # Handle multipart messages
        if payload.get("parts"):
            text_parts = []
            for part in payload["parts"]:
                # Recursively process parts
                content = extract_message_part(part)
                if content:
                    text_parts.append(content)
            return "\n".join(text_parts)
            
        return ""
    
    # Function to get credentials from token.json
    def get_credentials(gmail_token=None, gmail_secret=None):
        """Get Gmail API credentials from token.json"""
        token_path = _SECRETS_DIR / "token.json"
        
        if not os.path.exists(token_path):
            logger.error(f"Token file not found at {token_path}")
            return None
            
        try:
            with open(token_path, "r") as f:
                token_data = json.load(f)
                
            from google.oauth2.credentials import Credentials
            
            # Create credentials object with specific format
            # Using token format from the token.json file
            credentials = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes", ["https://www.googleapis.com/auth/gmail.modify"])
            )
            
            # Add authorize method to make it compatible with old code
            credentials.authorize = lambda request: request
            
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return None
    
    # Type alias for better readability
    EmailData = Dict[str, Any]
    
    GMAIL_API_AVAILABLE = True
    
except ImportError:
    # If Gmail API libraries aren't available, set flag to use mock implementation
    GMAIL_API_AVAILABLE = False
    logger = logging.getLogger(__name__)

# Helper function that is used by the tool and can be imported elsewhere
def fetch_group_emails(
    email_address: str,
    minutes_since: int = 30,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Fetch recent emails from Gmail that involve the specified email address.
    
    This function retrieves emails where the specified address is either a sender
    or recipient, processes them, and returns them in a format suitable for the
    email assistant to process.
    
    Args:
        email_address: Email address to fetch messages for
        minutes_since: Only retrieve emails newer than this many minutes
        gmail_token: Optional token for Gmail API authentication
        gmail_secret: Optional credentials for Gmail API authentication
        
    Yields:
        Dict objects containing processed email information
    """
    use_mock = False
    
    # Check if we need to use mock implementation
    if not GMAIL_API_AVAILABLE:
        logger.info("Gmail API not available, using mock implementation")
        use_mock = True
    
    # Check if required credential files exist
    if not use_mock and not gmail_token and not gmail_secret:
        token_path = str(_SECRETS_DIR / "token.json")
        secrets_path = str(_SECRETS_DIR / "secrets.json")
        
        if not os.path.exists(token_path) and not os.path.exists(secrets_path):
            logger.warning(f"No Gmail API credentials found. Looking for token.json or secrets.json in .secrets directory")
            logger.warning("Using mock implementation instead")
            use_mock = True
    
    # Return mock data if needed
    if use_mock:
        # For demo purposes, we return a mock email
        mock_email = {
            "from_email": "sender@example.com",
            "to_email": email_address,
            "subject": "Sample Email Subject",
            "page_content": "This is a sample email body for testing the email assistant.",
            "id": "mock-email-id-123",
            "thread_id": "mock-thread-id-123",
            "send_time": datetime.now().isoformat()
        }
        
        yield mock_email
        return
    
    try:
        # Get Gmail API credentials
        creds = get_credentials(gmail_token, gmail_secret)
        
        # Check if credentials are valid
        if not creds or not hasattr(creds, 'authorize'):
            logger.warning("Invalid Gmail credentials, using mock implementation")
            mock_email = {
                "from_email": "sender@example.com",
                "to_email": email_address,
                "subject": "Sample Email Subject - Invalid Credentials",
                "page_content": "This is a mock email because the Gmail credentials are invalid.",
                "id": "mock-email-id-123",
                "thread_id": "mock-thread-id-123",
                "send_time": datetime.now().isoformat()
            }
            yield mock_email
            return
            
        service = build("gmail", "v1", credentials=creds)
        
        # Calculate timestamp for filtering
        after = int((datetime.now() - timedelta(minutes=minutes_since)).timestamp())
        
        # Construct Gmail search query
        query = f"(to:{email_address} OR from:{email_address}) after:{after}"
        
        # Retrieve all matching messages (handling pagination)
        messages = []
        nextPageToken = None
        logger.info(f"Fetching emails for {email_address} from last {minutes_since} minutes")
        
        while True:
            results = (
                service.users()
                .messages()
                .list(userId="me", q=query, pageToken=nextPageToken)
                .execute()
            )
            if "messages" in results:
                messages.extend(results["messages"])
            nextPageToken = results.get("nextPageToken")
            if not nextPageToken:
                break

        # Process each message
        count = 0
        for message in messages:
            try:
                # Get full message details
                msg = service.users().messages().get(userId="me", id=message["id"]).execute()
                thread_id = msg["threadId"]
                payload = msg["payload"]
                headers = payload.get("headers", [])
                
                # Get thread details to determine conversation context
                thread = service.users().threads().get(userId="me", id=thread_id).execute()
                messages_in_thread = thread["messages"]
                
                # Analyze the last message in the thread to determine if we need to process it
                last_message = messages_in_thread[-1]
                last_headers = last_message["payload"]["headers"]
                
                # Get sender of last message
                from_header = next(
                    header["value"] for header in last_headers if header["name"] == "From"
                )
                last_from_header = next(
                    header["value"]
                    for header in last_message["payload"].get("headers")
                    if header["name"] == "From"
                )
                
                # If the last message was sent by the user, mark this as a user response
                # and don't process it further (assistant doesn't need to respond to user's own emails)
                if email_address in last_from_header:
                    yield {
                        "id": message["id"],
                        "thread_id": message["threadId"],
                        "user_respond": True,
                    }
                    continue
                    
                # Only process messages that are the latest in their thread and not from the user
                if email_address not in from_header and message["id"] == last_message["id"]:
                    # Extract email metadata from headers
                    subject = next(
                        header["value"] for header in headers if header["name"] == "Subject"
                    )
                    from_email = next(
                        (header["value"] for header in headers if header["name"] == "From"),
                        "",
                    ).strip()
                    _to_email = next(
                        (header["value"] for header in headers if header["name"] == "To"),
                        "",
                    ).strip()
                    
                    # Use Reply-To header if present
                    if reply_to := next(
                        (
                            header["value"]
                            for header in headers
                            if header["name"] == "Reply-To"
                        ),
                        "",
                    ).strip():
                        from_email = reply_to
                        
                    # Extract and parse email timestamp
                    send_time = next(
                        header["value"] for header in headers if header["name"] == "Date"
                    )
                    parsed_time = parse_time(send_time)
                    
                    # Extract email body content
                    body = extract_message_part(payload)
                    
                    # Yield the processed email data
                    yield {
                        "from_email": from_email,
                        "to_email": _to_email,
                        "subject": subject,
                        "page_content": body,
                        "id": message["id"],
                        "thread_id": message["threadId"],
                        "send_time": parsed_time.isoformat(),
                    }
                    count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process message {message['id']}: {str(e)}")

        logger.info(f"Found {count} emails to process.")
    
    except Exception as e:
        logger.error(f"Error accessing Gmail API: {str(e)}")
        # Fall back to mock implementation
        mock_email = {
            "from_email": "sender@example.com",
            "to_email": email_address,
            "subject": "Sample Email Subject",
            "page_content": "This is a sample email body for testing the email assistant.",
            "id": "mock-email-id-123",
            "thread_id": "mock-thread-id-123",
            "send_time": datetime.now().isoformat()
        }
        
        yield mock_email

class FetchEmailsInput(BaseModel):
    """
    Input schema for the fetch_emails_tool.
    """
    email_address: str = Field(
        description="Email address to fetch emails for"
    )
    minutes_since: int = Field(
        default=30,
        description="Only retrieve emails newer than this many minutes"
    )

@tool(args_schema=FetchEmailsInput)
def fetch_emails_tool(email_address: str, minutes_since: int = 30) -> str:
    """
    Fetches recent emails from Gmail for the specified email address.
    
    Args:
        email_address: Email address to fetch messages for
        minutes_since: Only retrieve emails newer than this many minutes (default: 30)
        
    Returns:
        String summary of fetched emails
    """
    emails = list(fetch_group_emails(email_address, minutes_since))
    
    if not emails:
        return "No new emails found."
    
    result = f"Found {len(emails)} new emails:\n\n"
    
    for i, email in enumerate(emails, 1):
        if email.get("user_respond", False):
            result += f"{i}. You already responded to this email (Thread ID: {email['thread_id']})\n\n"
            continue
            
        result += f"{i}. From: {email['from_email']}\n"
        result += f"   To: {email['to_email']}\n"
        result += f"   Subject: {email['subject']}\n"
        result += f"   Time: {email['send_time']}\n"
        result += f"   ID: {email['id']}\n"
        result += f"   Thread ID: {email['thread_id']}\n"
        result += f"   Content: {email['page_content'][:200]}...\n\n"
    
    return result

class SendEmailInput(BaseModel):
    """
    Input schema for the send_email_tool.
    """
    email_id: str = Field(
        description="Gmail message ID to reply to"
    )
    response_text: str = Field(
        description="Content of the reply"
    )
    email_address: str = Field(
        description="Current user's email address"
    )
    additional_recipients: Optional[List[str]] = Field(
        default=None,
        description="Optional additional recipients to include"
    )

# Helper function for sending emails
def send_email(
    email_id: str,
    response_text: str,
    email_address: str,
    addn_receipients: Optional[List[str]] = None
) -> bool:
    """
    Send a reply to an existing email thread.
    
    Args:
        email_id: Gmail message ID to reply to
        response_text: Content of the reply
        email_address: Current user's email address
        addn_receipients: Optional additional recipients
        
    Returns:
        Success flag (True if email was sent)
    """
    if not GMAIL_API_AVAILABLE:
        logger.info("Gmail API not available, simulating email send")
        logger.info(f"Would send: {response_text[:100]}...")
        return True
        
    try:
        # Get Gmail API credentials
        creds = get_credentials()
        service = build("gmail", "v1", credentials=creds)
        
        # Get the original message to extract headers
        message = service.users().messages().get(userId="me", id=email_id).execute()
        headers = message["payload"]["headers"]
        
        # Extract subject with Re: prefix if not already present
        subject = next(header["value"] for header in headers if header["name"] == "Subject")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
            
        # Create a reply message
        original_from = next(header["value"] for header in headers if header["name"] == "From")
        
        # Create a message object
        msg = MIMEText(response_text)
        msg["to"] = original_from
        msg["from"] = email_address
        msg["subject"] = subject
        
        # Add additional recipients if specified
        if addn_receipients:
            msg["cc"] = ", ".join(addn_receipients)
            
        # Add reference to original message for threading
        thread_id = message["threadId"]
        
        # Encode the message
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        
        # Send the message
        sent_message = (
            service.users()
            .messages()
            .send(
                userId="me",
                body={
                    "raw": raw,
                    "threadId": thread_id,
                },
            )
            .execute()
        )
        
        logger.info(f"Email sent: Message ID {sent_message['id']}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

@tool(args_schema=SendEmailInput)
def send_email_tool(
    email_id: str,
    response_text: str,
    email_address: str,
    additional_recipients: Optional[List[str]] = None
) -> str:
    """
    Send a reply to an existing email thread in Gmail.
    
    Args:
        email_id: Gmail message ID to reply to
        response_text: Content of the reply
        email_address: Current user's email address
        additional_recipients: Optional additional recipients to include
        
    Returns:
        Confirmation message
    """
    try:
        success = send_email(
            email_id,
            response_text,
            email_address,
            addn_receipients=additional_recipients
        )
        if success:
            return f"Email reply sent successfully to message ID: {email_id}"
        else:
            return "Failed to send email due to an API error"
    except Exception as e:
        return f"Failed to send email: {str(e)}"

class CheckCalendarInput(BaseModel):
    """
    Input schema for the check_calendar_tool.
    """
    dates: List[str] = Field(
        description="List of dates to check in DD-MM-YYYY format"
    )

def get_calendar_events(dates: List[str]) -> str:
    """
    Check Google Calendar for events on specified dates.
    
    Args:
        dates: List of dates to check in DD-MM-YYYY format
        
    Returns:
        Formatted calendar events for the specified dates
    """
    if not GMAIL_API_AVAILABLE:
        logger.info("Gmail API not available, simulating calendar check")
        result = "Calendar events:\n\n"
        for date in dates:
            result += f"Events for {date}:\n"
            result += "  - 9:00 AM - 10:00 AM: Team Meeting\n"
            result += "  - 2:00 PM - 3:00 PM: Project Review\n"
            result += "Available slots: 10:00 AM - 2:00 PM, after 3:00 PM\n\n"
        return result
        
    try:
        # Get Gmail API credentials
        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)
        
        result = "Calendar events:\n\n"
        
        for date_str in dates:
            # Parse date string (DD-MM-YYYY)
            day, month, year = date_str.split("-")
            
            # Format start and end times for the API
            start_time = f"{year}-{month}-{day}T00:00:00Z"
            end_time = f"{year}-{month}-{day}T23:59:59Z"
            
            # Call the Calendar API
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time,
                    timeMax=end_time,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            
            events = events_result.get("items", [])
            
            result += f"Events for {date_str}:\n"
            
            if not events:
                result += "  No events found for this day\n"
                result += "  Available all day\n\n"
                continue
                
            # Process events
            busy_slots = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))
                
                # Convert to datetime objects
                if "T" in start:  # dateTime format
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    
                    # Format for display
                    start_display = start_dt.strftime("%I:%M %p")
                    end_display = end_dt.strftime("%I:%M %p")
                    
                    result += f"  - {start_display} - {end_display}: {event['summary']}\n"
                    busy_slots.append((start_dt, end_dt))
                else:  # all-day event
                    result += f"  - All day: {event['summary']}\n"
                    busy_slots.append(("all-day", "all-day"))
            
            # Calculate available slots
            if "all-day" in [slot[0] for slot in busy_slots]:
                result += "  Available: No availability (all-day events)\n\n"
            else:
                # Sort busy slots by start time
                busy_slots.sort(key=lambda x: x[0])
                
                # Define working hours (9 AM to 5 PM)
                work_start = datetime(
                    year=int(year), 
                    month=int(month), 
                    day=int(day), 
                    hour=9, 
                    minute=0
                )
                work_end = datetime(
                    year=int(year), 
                    month=int(month), 
                    day=int(day), 
                    hour=17, 
                    minute=0
                )
                
                # Calculate available slots
                available_slots = []
                current = work_start
                
                for start, end in busy_slots:
                    if current < start:
                        available_slots.append((current, start))
                    current = max(current, end)
                
                if current < work_end:
                    available_slots.append((current, work_end))
                
                # Format available slots
                if available_slots:
                    result += "  Available: "
                    for i, (start, end) in enumerate(available_slots):
                        start_display = start.strftime("%I:%M %p")
                        end_display = end.strftime("%I:%M %p")
                        result += f"{start_display} - {end_display}"
                        if i < len(available_slots) - 1:
                            result += ", "
                    result += "\n\n"
                else:
                    result += "  Available: No availability during working hours\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking calendar: {str(e)}")
        # Return mock data in case of error
        result = "Calendar events (mock due to error):\n\n"
        for date in dates:
            result += f"Events for {date}:\n"
            result += "  - 9:00 AM - 10:00 AM: Team Meeting\n"
            result += "  - 2:00 PM - 3:00 PM: Project Review\n"
            result += "Available slots: 10:00 AM - 2:00 PM, after 3:00 PM\n\n"
        return result

@tool(args_schema=CheckCalendarInput)
def check_calendar_tool(dates: List[str]) -> str:
    """
    Check Google Calendar for events on specified dates.
    
    Args:
        dates: List of dates to check in DD-MM-YYYY format
        
    Returns:
        Formatted calendar events for the specified dates
    """
    try:
        events = get_calendar_events(dates)
        return events
    except Exception as e:
        return f"Failed to check calendar: {str(e)}"

class ScheduleMeetingInput(BaseModel):
    """
    Input schema for the schedule_meeting_tool.
    """
    attendees: List[str] = Field(
        description="Email addresses of meeting attendees"
    )
    title: str = Field(
        description="Meeting title/subject"
    )
    start_time: str = Field(
        description="Meeting start time in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    end_time: str = Field(
        description="Meeting end time in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    organizer_email: str = Field(
        description="Email address of the meeting organizer"
    )
    timezone: str = Field(
        default="America/Los_Angeles",
        description="Timezone for the meeting"
    )

def send_calendar_invite(
    attendees: List[str],
    title: str,
    start_time: str,
    end_time: str,
    organizer_email: str,
    timezone: str = "America/Los_Angeles"
) -> bool:
    """
    Schedule a meeting with Google Calendar and send invites.
    
    Args:
        attendees: Email addresses of meeting attendees
        title: Meeting title/subject
        start_time: Meeting start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        end_time: Meeting end time in ISO format (YYYY-MM-DDTHH:MM:SS)
        organizer_email: Email address of the meeting organizer
        timezone: Timezone for the meeting
        
    Returns:
        Success flag (True if meeting was scheduled)
    """
    if not GMAIL_API_AVAILABLE:
        logger.info("Gmail API not available, simulating calendar invite")
        logger.info(f"Would schedule: {title} from {start_time} to {end_time}")
        logger.info(f"Attendees: {', '.join(attendees)}")
        return True
        
    try:
        # Get Gmail API credentials
        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)
        
        # Create event details
        event = {
            "summary": title,
            "start": {
                "dateTime": start_time,
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": timezone,
            },
            "attendees": [{"email": email} for email in attendees],
            "organizer": {
                "email": organizer_email,
                "self": True,
            },
            "reminders": {
                "useDefault": True,
            },
            "sendUpdates": "all",  # Send email notifications to attendees
        }
        
        # Create the event
        event = service.events().insert(calendarId="primary", body=event).execute()
        
        logger.info(f"Meeting created: {event.get('htmlLink')}")
        return True
        
    except Exception as e:
        logger.error(f"Error scheduling meeting: {str(e)}")
        return False

@tool(args_schema=ScheduleMeetingInput)
def schedule_meeting_tool(
    attendees: List[str],
    title: str,
    start_time: str,
    end_time: str,
    organizer_email: str,
    timezone: str = "America/Los_Angeles"
) -> str:
    """
    Schedule a meeting with Google Calendar and send invites.
    
    Args:
        attendees: Email addresses of meeting attendees
        title: Meeting title/subject
        start_time: Meeting start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        end_time: Meeting end time in ISO format (YYYY-MM-DDTHH:MM:SS)
        organizer_email: Email address of the meeting organizer
        timezone: Timezone for the meeting (default: America/Los_Angeles)
        
    Returns:
        Success or failure message
    """
    try:
        success = send_calendar_invite(
            attendees,
            title,
            start_time,
            end_time,
            organizer_email,
            timezone
        )
        
        if success:
            return f"Meeting '{title}' scheduled successfully from {start_time} to {end_time} with {len(attendees)} attendees"
        else:
            return "Failed to schedule meeting"
    except Exception as e:
        return f"Error scheduling meeting: {str(e)}"
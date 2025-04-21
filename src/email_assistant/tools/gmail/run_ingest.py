#!/usr/bin/env python
"""
Gmail ingestion script for Email Assistant.

This script fetches recent emails from Gmail and processes them through
the email assistant LangGraph. It can be run on a schedule to continuously monitor
and process new emails.

Example usage:
  python src/email_assistant/tools/gmail/run_ingest.py --email your.email@gmail.com --minutes-since 60
"""

import os
import sys
import argparse
import asyncio
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Import Gmail tools
from src.email_assistant.tools.gmail.gmail_tools import fetch_group_emails

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Try to import LangGraph SDK
    from langgraph_sdk import get_client
    import httpx
    LANGGRAPH_SDK_AVAILABLE = True
except ImportError:
    logger.warning("LangGraph SDK not available. Running in mock mode only.")
    LANGGRAPH_SDK_AVAILABLE = False


async def process_emails(args):
    """Process emails from Gmail using the Email Assistant."""
    
    # Validate email address
    if not args.email:
        logger.error("Email address is required. Use --email or set EMAIL_ADDRESS env var.")
        return 1
        
    # Create log directory
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Fetch emails
    logger.info(f"Fetching emails for {args.email} from the last {args.minutes_since} minutes...")
    
    # Check if we're running in mock mode
    if args.mock or not LANGGRAPH_SDK_AVAILABLE:
        if not LANGGRAPH_SDK_AVAILABLE:
            logger.warning("LangGraph SDK not available - running in mock mode")
        else:
            logger.info("Mock mode enabled - simulating LangGraph processing")
            
        # Iterate through emails but don't actually process them
        email_count = 0
        try:
            for email in fetch_group_emails(
                args.email,
                minutes_since=args.minutes_since,
                gmail_token=args.gmail_token,
                gmail_secret=args.gmail_secret,
                include_read=args.include_read,
                skip_filters=args.skip_filters
            ):
                email_count += 1
                logger.info(f"Would process email from {email['from_email']} with subject: {email['subject']}")
                if args.early and email_count > 0:
                    logger.info("Early stop enabled, stopping after first email")
                    break
            logger.info(f"Found {email_count} emails to process (mock mode)")
            
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            return 1
            
        return 0
        
    # Set up direct REST API interaction mode if needed
    # This is a fallback if the LangGraph SDK connection fails
    api_mode = False
    import requests
    
    # Initialize LangGraph client
    try:
        client = get_client(url=args.url)
        # Test connection by making a simple request - use the correct SDK API structure
        try:
            # LangGraph SDK structure might differ depending on the version
            # Try common operations to see if the client works
            try:
                # Just check if the client has essential methods
                if hasattr(client, 'runs') and hasattr(client, 'threads'):
                    logger.info(f"Connected to LangGraph server at {args.url}")
                else:
                    logger.warning("LangGraph client initialized but missing required methods")
                    api_mode = True
            except Exception:
                logger.warning("Unable to validate LangGraph client structure")
                api_mode = True
                
        except Exception as conn_error:
            logger.warning(f"Could not communicate with LangGraph SDK API: {str(conn_error)}")
            logger.warning("Switching to direct REST API mode")
            api_mode = True
            
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph client for {args.url}: {str(e)}")
        
        # Try a direct HTTP request to see if the server is running but has a different API
        try:
            direct_urls = [
                f"{args.url}/health",
                f"{args.url}/v1/health",
                f"{args.url}/api/v1/health"
            ]
            server_running = False
            
            for url in direct_urls:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        server_running = True
                        logger.warning(f"LangGraph server is running at {url}")
                        break
                except Exception:
                    pass
            
            if server_running:
                logger.warning("Switching to direct REST API mode")
                api_mode = True
            else:
                raise Exception("Server not responding correctly")
                
        except Exception:
            logger.error("\nPlease start the LangGraph server by running the following command in a separate terminal:")
            logger.error("  cd /Users/rlm/Desktop/Code/interrupt_workshop && langgraph start")
            logger.error("\nOr to run with mock responses instead, use the --mock flag:")
            logger.error(f"  python src/email_assistant/tools/gmail/run_ingest.py --email {args.email} --mock\n")
            return 1
    
    # Process emails
    processed_count = 0
    
    try:
        for email in fetch_group_emails(
            args.email,
            minutes_since=args.minutes_since,
            gmail_token=args.gmail_token,
            gmail_secret=args.gmail_secret,
            include_read=args.include_read,
            skip_filters=args.skip_filters
        ):
            # Generate a consistent thread ID using MD5 hash
            thread_id = str(
                uuid.UUID(hex=hashlib.md5(email["thread_id"].encode("UTF-8")).hexdigest())
            )
            
            # Different handling depending on whether we're using SDK or direct REST API
            if api_mode:
                # Direct REST API mode
                try:
                    # Process the email directly with REST API
                    logger.info(f"Processing email from: {email['from_email']}")
                    logger.info(f"Subject: {email['subject']}")
                    
                    # Prepare the payload for LangGraph using direct API schema
                    # Using more intuitive field names
                    payload = {
                        "inputs": {
                            "email_input": {
                                "from": email["from_email"],
                                "to": email["to_email"],
                                "subject": email["subject"],
                                "body": email["page_content"]
                            }
                        }
                    }
                    
                    # Try different API URL patterns
                    api_urls = [
                        f"{args.url}/v1/graphs/{args.graph_name}",
                        f"{args.url}/api/v1/graphs/{args.graph_name}",
                        f"{args.url}/v1/invoke/{args.graph_name}",
                        f"{args.url}/v1/{args.graph_name}"
                    ]
                    
                    success = False
                    result = None
                    
                    for api_url in api_urls:
                        try:
                            logger.info(f"Trying direct API call to {api_url}")
                            response = requests.post(
                                api_url, 
                                json=payload,
                                timeout=10
                            )
                            
                            if response.status_code in (200, 201, 202):
                                success = True
                                result = response
                                logger.info(f"Successful API call to {api_url}")
                                break
                                
                        except Exception as url_err:
                            logger.warning(f"API call to {api_url} failed: {str(url_err)}")
                            
                    # Use the last successful response or the last tried one
                    response = result if success else response
                    
                    if response.status_code == 200:
                        logger.info(f"Successfully processed email with ID: {email['id']}")
                        processed_count += 1
                    else:
                        logger.error(f"Error processing email: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    logger.error(f"Error in direct API mode: {str(e)}")
            else:
                # SDK mode
                try:
                    # Try to get existing thread info
                    thread_info = await client.threads.get(thread_id)
                    logger.info(f"Found existing thread: {thread_id}")
                except httpx.HTTPStatusError as e:
                    # If the user already responded to this email, skip it
                    if "user_respond" in email:
                        logger.info(f"Skipping email {email.get('id', '')}: User already responded")
                        continue
                        
                    # If thread doesn't exist, create it
                    if e.response.status_code == 404:
                        logger.info(f"Creating new thread: {thread_id}")
                        thread_info = await client.threads.create(thread_id=thread_id)
                    else:
                        logger.error(f"HTTP error: {str(e)}")
                        raise e
                        
                # If the user already responded to this email, mark thread as complete and skip
                if "user_respond" in email:
                    logger.info(f"User already responded to email {email.get('id', '')}, marking thread as complete")
                    await client.threads.update_state(thread_id, None, as_node="__end__")
                    continue
                    
                # Check if we've already processed this email
                recent_email = thread_info["metadata"].get("email_id")
                if recent_email == email["id"]:
                    if args.early:
                        logger.info("Encountered already processed email, early stop enabled")
                        break
                    elif not args.rerun:
                        logger.info(f"Already processed email {email['id']}, skipping")
                        continue
                        
                # Update thread metadata with current email ID
                await client.threads.update(thread_id, metadata={"email_id": email["id"]})
                
                # Log email details
                logger.info(f"Processing email from: {email['from_email']}")
                logger.info(f"Subject: {email['subject']}")
                
                # Create a run for this email
                try:
                    logger.info(f"Creating run for thread {thread_id} with graph {args.graph_name}")
                    await client.runs.create(
                        thread_id,
                        args.graph_name,
                        input={"email_input": {
                            "from": email["from_email"],
                            "to": email["to_email"],
                            "subject": email["subject"],
                            "body": email["page_content"]
                        }},
                        multitask_strategy="rollback",
                    )
                    logger.info(f"Successfully processed email with ID: {email['id']}")
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing email: {str(e)}")
                
            # Early stop after processing one email if requested
            if args.early and processed_count > 0:
                logger.info("Early stop enabled, stopping after first email")
                break
                
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error fetching or processing emails: {error_message}")
        
        # Check for connection errors specifically
        if "connection" in error_message.lower() or "connect" in error_message.lower():
            logger.error("\n💡 Connection Error: The LangGraph server is not running!")
            logger.error("\nTo fix this, you have two options:")
            logger.error("1. Start the LangGraph server in a new terminal window:")
            logger.error("   cd /Users/rlm/Desktop/Code/interrupt_workshop && langgraph start")
            logger.error("\n2. Use mock mode to test without a server:")
            logger.error(f"   python src/email_assistant/tools/gmail/run_ingest.py --email {args.email} --mock")
        
        return 1
            
    logger.info(f"Email processing complete. Processed {processed_count} emails.")
    return 0
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gmail ingestion script for Email Assistant")
    parser.add_argument(
        "--email", 
        type=str, 
        default=os.environ.get("EMAIL_ADDRESS"),
        help="Email address to fetch messages for (can also set EMAIL_ADDRESS env var)"
    )
    parser.add_argument(
        "--minutes-since", 
        type=int, 
        default=60,
        help="Only retrieve emails newer than this many minutes"
    )
    parser.add_argument(
        "--graph-name", 
        type=str, 
        default="email_assistant_hitl_memory",
        help="Name of the LangGraph to use"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://127.0.0.1:2024",
        help="URL of the LangGraph deployment"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="email_logs",
        help="Directory to store email logs"
    )
    parser.add_argument(
        "--rerun", 
        type=int, 
        default=0,
        help="Process the same emails for testing (1=yes, 0=no)"
    )
    parser.add_argument(
        "--early", 
        type=int, 
        default=0,
        help="Early stop after processing one email (1=yes, 0=no)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without requiring a LangGraph server"
    )
    parser.add_argument(
        "--include-read",
        action="store_true",
        help="Include emails that have already been read"
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip filtering of emails (include messages that would normally be filtered out)"
    )
    parser.add_argument(
        "--gmail-token",
        type=str,
        default=None,
        help="The token to use in communicating with the Gmail API"
    )
    parser.add_argument(
        "--gmail-secret",
        type=str,
        default=None,
        help="The credentials to use in communicating with the Gmail API"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if LANGGRAPH_SDK_AVAILABLE:
        exit(asyncio.run(process_emails(args)))
    else:
        # If LangGraph SDK isn't available, run synchronously in mock mode
        args.mock = True
        exit(process_emails(args))
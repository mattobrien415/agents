import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Optional
# Replace eaia imports with local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader.gmail.gmail import fetch_group_emails
from src.email_assistant.schemas import EmailData
from langgraph_sdk import get_client
import httpx
import uuid
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(
    url: Optional[str] = None,
    minutes_since: int = 60,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
    early: bool = True,
    rerun: bool = False,
    email: Optional[str] = None,
    log_dir: str = "email_logs",
):
    # Use the provided email or get from environment variable
    if email is None:
        email_address = os.environ.get("EMAIL_ADDRESS")
        if email_address is None:
            raise ValueError("No email address provided. Set EMAIL_ADDRESS environment variable or use --email argument.")
    else:
        email_address = email
    if url is None:
        # Connect to local LangGraph server
        client = get_client(url="http://127.0.0.1:2024")
        # Set the graph to email_assistant_hitl_memory
        client.graph = "email_assistant_hitl_memory"
    else:
        client = get_client(
            url=url
        )
        # Set the graph to email_assistant_hitl_memory
        client.graph = "email_assistant_hitl_memory"
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"email_dump_{timestamp}.json")
    logger.info(f"Will dump emails to {log_file}")
    
    # List to store emails for logging
    emails_to_log = []

    # TODO: This really should be async
    for email in fetch_group_emails(
        email_address,
        minutes_since=minutes_since,
        gmail_token=gmail_token,
        gmail_secret=gmail_secret,
    ):
        # Add email to log list
        emails_to_log.append(email)
        thread_id = str(
            uuid.UUID(hex=hashlib.md5(email["thread_id"].encode("UTF-8")).hexdigest())
        )
        try:
            thread_info = await client.threads.get(thread_id)
        except httpx.HTTPStatusError as e:
            if "user_respond" in email:
                continue
            if e.response.status_code == 404:
                thread_info = await client.threads.create(thread_id=thread_id)
            else:
                raise e
        if "user_respond" in email:
            await client.threads.update_state(thread_id, None, as_node="__end__")
            continue
        recent_email = thread_info["metadata"].get("email_id")
        if recent_email == email["id"]:
            if early:
                break
            else:
                if rerun:
                    pass
                else:
                    continue
        await client.threads.update(thread_id, metadata={"email_id": email["id"]})

        await client.runs.create(
            thread_id,
            "email_assistant",
            input={"email_input": email},
            multitask_strategy="rollback",
        )
    
    # Write all emails to log file
    if emails_to_log:
        logger.info(f"Writing {len(emails_to_log)} emails to {log_file}")
        with open(log_file, "w") as f:
            json.dump(emails_to_log, f, indent=2, default=str)
        logger.info(f"Email dump completed successfully to {log_file}")
    else:
        logger.info("No emails collected during this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to run against",
    )
    parser.add_argument(
        "--early",
        type=int,
        default=1,
        help="whether to break when encountering seen emails",
    )
    parser.add_argument(
        "--rerun",
        type=int,
        default=0,
        help="whether to rerun all emails",
    )
    parser.add_argument(
        "--minutes-since",
        type=int,
        default=60,
        help="Only process emails that are less than this many minutes old.",
    )
    parser.add_argument(
        "--gmail-token",
        type=str,
        default=None,
        help="The token to use in communicating with the Gmail API.",
    )
    parser.add_argument(
        "--gmail-secret",
        type=str,
        default=None,
        help="The creds to use in communicating with the Gmail API.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="The email address to use",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="email_logs",
        help="Directory to store email logs",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            url=args.url,
            minutes_since=args.minutes_since,
            gmail_token=args.gmail_token,
            gmail_secret=args.gmail_secret,
            early=bool(args.early),
            rerun=bool(args.rerun),
            email=args.email,
            log_dir=args.log_dir,
        )
    )

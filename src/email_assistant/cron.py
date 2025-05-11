import os
import sys
import asyncio
from typing import Dict, Any, TypedDict
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, START, END

# Ensure path includes the project directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from email_assistant.tools.gmail.run_ingest import fetch_and_process_emails

@dataclass(kw_only=True)
class JobKickoff:
    """State for the email ingestion cron job"""
    email: str
    minutes_since: int = 60
    graph_name: str = "email_assistant_hitl_memory"
    url: str = "http://127.0.0.1:2024"
    include_read: bool = False
    rerun: bool = False
    early: bool = False
    skip_filters: bool = False

async def main(state: JobKickoff):
    """Run the email ingestion process"""
    print(f"Kicking off job to fetch emails from the past {state.minutes_since} minutes")
    
    # Convert state to args object for fetch_and_process_emails
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(
        email=state.email,
        minutes_since=state.minutes_since,
        graph_name=state.graph_name,
        url=state.url,
        include_read=state.include_read,
        rerun=state.rerun,
        early=state.early,
        skip_filters=state.skip_filters
    )
    
    # Run the ingestion process
    result = await fetch_and_process_emails(args)
    
    # Return the result status
    return {"status": "success" if result == 0 else "error", "exit_code": result}

# Build the graph
graph = StateGraph(JobKickoff)
graph.add_node("ingest_emails", main)
graph.set_entry_point("ingest_emails")
graph = graph.compile()
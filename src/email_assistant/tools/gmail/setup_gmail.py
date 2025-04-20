#!/usr/bin/env python
"""
Setup script for Gmail API integration.

This script handles the OAuth flow for Gmail API access by:
1. Creating a .secrets directory if it doesn't exist
2. Using credentials from .secrets/secrets.json to authenticate
3. Opening a browser window for user authentication
4. Storing the access token in .secrets/token.json
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from src.email_assistant.tools.gmail.gmail_tools import get_credentials

def main():
    """Run Gmail authentication setup."""
    # Create .secrets directory
    secrets_dir = Path(__file__).parent.absolute() / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for secrets.json
    secrets_path = secrets_dir / "secrets.json"
    if not secrets_path.exists():
        print(f"Error: Client secrets file not found at {secrets_path}")
        print("Please download your OAuth client ID JSON from Google Cloud Console")
        print("and save it as .secrets/secrets.json")
        return 1
    
    print("Starting Gmail API authentication flow...")
    print("A browser window will open for you to authorize access.")
    
    # This will trigger the OAuth flow and create token.json
    try:
        get_credentials()
        print("\nAuthentication successful!")
        print(f"Access token stored at {secrets_dir / 'token.json'}")
        return 0
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
"""
Custom tools for the MCP chatbot.
These are additional tools that can be used alongside MCP server tools.
Includes Gmail and Google Calendar management tools.
"""

import os
import json
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime, timedelta, timezone
from typing import List
from langchain_core.tools import StructuredTool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

# Email Configuration
GMAIL_USER = os.getenv('USER_GOOGLE_EMAIL', 'buildbot312@gmail.com')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')

# Calendar OAuth Scopes
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']


# ==================== HELPER FUNCTIONS ====================

def get_calendar_credentials():
    """
    Authenticate and retrieve Google Calendar API credentials.
    
    This function handles the OAuth2 flow for Google Calendar API access.
    It first checks for existing credentials in 'calendar_token.json'.
    Once authenticated, credentials are saved and reused automatically.
    
    Returns:
        Credentials: Google OAuth2 credentials object for Calendar API access
        
    Raises:
        Exception: If credentials cannot be created
    """
    creds = None
    
    # Determine token file path (use backend directory)
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    token_file = os.path.join(backend_dir, 'calendar_token.json')
    
    # Try to load existing credentials
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, CALENDAR_SCOPES)
        except Exception as e:
            print(f"Warning: Could not load token file: {e}")
            creds = None
    
    # Check if credentials are valid
    if creds and creds.valid:
        return creds
    
    # Try to refresh if expired
    if creds and creds.expired and creds.refresh_token:
        try:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
            # Save refreshed credentials
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
            return creds
        except Exception as e:
            print(f"Warning: Could not refresh token: {e}")
            creds = None
    
    # Need to authenticate - look for client secret file or env vars
    client_secret_file = 'client_secret_447640181348-pe428bqnghc4ee0ra031upncig6o7fnm.apps.googleusercontent.com.json'
    
    # Check in backend directory
    client_secret_path = os.path.join(backend_dir, client_secret_file)
    
    if os.path.exists(client_secret_path):
        print(f"Using client secret file: {client_secret_path}")
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secret_path,
            CALENDAR_SCOPES
        )
    else:
        # Try to create from environment variables
        client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        client_secret = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
        
        if client_id and client_secret:
            print("Using credentials from environment variables...")
            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost:3001/", "urn:ietf:wg:oauth:2.0:oob"]
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, CALENDAR_SCOPES)
        else:
            raise FileNotFoundError(
                f"Google OAuth credentials not found. Please either:\n"
                f"1. Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in .env file, OR\n"
                f"2. Place client secret JSON in: {client_secret_path}"
            )
    
    # Run OAuth flow (this will open browser ONCE)
    print("\n" + "="*80)
    print("🔐 FIRST TIME SETUP - Google Calendar Authentication")
    print("="*80)
    print("A browser window will open for you to authorize access.")
    print("After authorization, credentials will be saved and reused automatically.")
    print("You won't need to do this again unless you revoke access.")
    print("="*80 + "\n")
    
    try:
        # Try with port 3001 first
        try:
            creds = flow.run_local_server(
                port=3001, 
                open_browser=True,
                access_type='offline',
                prompt='consent',
                success_message='Authorization successful! You can close this window.',
                bind_addr='127.0.0.1'
            )
        except Exception as e:
            print(f"Port 3001 failed, trying 8080... Error: {e}")
            # Fallback to port 8080
            creds = flow.run_local_server(
                port=8080, 
                open_browser=True,
                access_type='offline',
                prompt='consent',
                success_message='Authorization successful! You can close this window.',
                bind_addr='127.0.0.1'
            )
    except Exception as e:
        print(f"\n❌ OAuth flow failed: {e}")
        print("\nTrying alternative method...")
        # If local server fails, use console flow
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        print(f"\nPlease visit this URL to authorize:\n{auth_url}\n")
        code = input("Enter the authorization code: ")
        flow.fetch_token(code=code)
        creds = flow.credentials
    
    # Save token for future use
    with open(token_file, 'w') as token:
        token.write(creds.to_json())
    
    print("\n✅ Authentication successful! Credentials saved.")
    print(f"Token saved to: {token_file}")
    print("You won't need to authenticate again.\n")
    
    return creds


def connect_imap():
    """
    Establish a connection to Gmail IMAP server.
    
    Creates and authenticates an IMAP4_SSL connection to Gmail's IMAP server
    using the configured email and app password.
    
    Returns:
        imaplib.IMAP4_SSL: Authenticated IMAP connection object
        
    Raises:
        Exception: If GMAIL_APP_PASSWORD is not set in environment variables
    """
    if not GMAIL_APP_PASSWORD:
        raise Exception("GMAIL_APP_PASSWORD not set in .env file")
    
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
    return mail


# ==================== GMAIL FUNCTIONS ====================

def send_email(to: str, subject: str, message: str, cc: str = "", bcc: str = ""):
    """
    Send an email using Gmail SMTP with App Password authentication.
    
    This function sends an email from the configured Gmail account using SMTP
    over SSL. It supports CC and BCC recipients.
    
    Args:
        to (str): Primary recipient email address
        subject (str): Email subject line
        message (str): Plain text email body content
        cc (str, optional): Comma-separated CC email addresses. Defaults to ""
        bcc (str, optional): Comma-separated BCC email addresses. Defaults to ""
        
    Returns:
        str: Success message with recipient email or error message
        
    Example:
        >>> send_email("user@example.com", "Hello", "This is a test email")
        "✅ Email sent successfully to user@example.com!"
    """
    try:
        if not GMAIL_APP_PASSWORD:
            return "❌ Error: GMAIL_APP_PASSWORD not set in .env file"
        
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = to
        msg['Subject'] = subject
        if cc:
            msg['Cc'] = cc
        if bcc:
            msg['Bcc'] = bcc
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Connect to Gmail SMTP
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        
        recipients = [to]
        if cc:
            recipients.extend([e.strip() for e in cc.split(',')])
        if bcc:
            recipients.extend([e.strip() for e in bcc.split(',')])
        
        server.send_message(msg)
        server.quit()
        
        return f"✅ Email sent successfully to {to}!"
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"


def list_emails(limit: int = 5, folder: str = "INBOX", unread_only: bool = False):
    """
    Retrieve a list of emails from Gmail using IMAP.
    
    Fetches emails from the specified folder and returns their metadata including
    subject, sender, date, and a snippet of the body. Results are ordered from
    most recent to oldest.
    
    Args:
        limit (int, optional): Maximum number of emails to retrieve. Defaults to 5
        folder (str, optional): Gmail folder/label to search. Defaults to "INBOX"
        unread_only (bool, optional): If True, only fetch unread emails. Defaults to False
        
    Returns:
        str: JSON string containing list of email objects or "No emails found." message
        Each email object contains: id, subject, from, date, snippet
        
    Example:
        >>> list_emails(limit=3, unread_only=True)
        '[{"id": "123", "subject": "Test", "from": "user@example.com", ...}]'
    """
    try:
        mail = connect_imap()
        mail.select(folder)
        
        search_criteria = 'UNSEEN' if unread_only else 'ALL'
        _, messages = mail.search(None, search_criteria)
        
        email_ids = messages[0].split()
        email_ids.reverse()  # Get most recent first
        email_ids = email_ids[:limit]
        
        emails = []
        for email_id in email_ids:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            subject = decode_header(email_message['Subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            from_ = email_message.get('From')
            date = email_message.get('Date')
            
            # Get email body
            body = ""
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = email_message.get_payload(decode=True).decode()
            
            emails.append({
                'id': email_id.decode(),
                'subject': subject,
                'from': from_,
                'date': date,
                'snippet': body[:200] if body else ""
            })
        
        mail.close()
        mail.logout()
        
        return json.dumps(emails, indent=2) if emails else "No emails found."
    except Exception as e:
        return f"❌ Error listing emails: {str(e)}"


def read_email(email_id: str, folder: str = "INBOX"):
    """
    Read the full content of a specific email by its ID.
    
    Fetches and parses the complete email message including headers and body
    content. Returns all relevant email metadata and the full message body.
    
    Args:
        email_id (str): The unique IMAP message ID of the email to read
        folder (str, optional): Gmail folder containing the email. Defaults to "INBOX"
        
    Returns:
        str: JSON string containing email details (subject, from, to, date, body)
        or error message if email cannot be retrieved
        
    Example:
        >>> read_email("12345")
        '{"subject": "Meeting", "from": "boss@company.com", "body": "..."}'
    """
    try:
        mail = connect_imap()
        mail.select(folder)
        
        _, msg_data = mail.fetch(email_id.encode(), '(RFC822)')
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        subject = decode_header(email_message['Subject'])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        
        from_ = email_message.get('From')
        to = email_message.get('To')
        date = email_message.get('Date')
        
        # Get email body
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()
        
        mail.close()
        mail.logout()
        
        email_data = {
            'subject': subject,
            'from': from_,
            'to': to,
            'date': date,
            'body': body
        }
        
        return json.dumps(email_data, indent=2)
    except Exception as e:
        return f"❌ Error reading email: {str(e)}"


def search_emails(query: str, limit: int = 10):
    """
    Search for emails matching a query string in their subject line.
    
    Performs a simple IMAP search for emails with the query text in the subject.
    Returns basic metadata for matching emails.
    
    Args:
        query (str): Search term to find in email subjects
        limit (int, optional): Maximum number of results to return. Defaults to 10
        
    Returns:
        str: JSON string containing list of matching emails or "No emails found" message
        Each email contains: id, subject, from, date
        
    Example:
        >>> search_emails("invoice", limit=5)
        '[{"id": "123", "subject": "Invoice #123", "from": "billing@company.com"}]'
    """
    try:
        mail = connect_imap()
        mail.select('INBOX')
        
        # Simple search by subject
        _, messages = mail.search(None, f'(SUBJECT "{query}")')
        
        email_ids = messages[0].split()
        email_ids.reverse()
        email_ids = email_ids[:limit]
        
        emails = []
        for email_id in email_ids:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            subject = decode_header(email_message['Subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            emails.append({
                'id': email_id.decode(),
                'subject': subject,
                'from': email_message.get('From'),
                'date': email_message.get('Date')
            })
        
        mail.close()
        mail.logout()
        
        return json.dumps(emails, indent=2) if emails else f"No emails found matching '{query}'."
    except Exception as e:
        return f"❌ Error searching emails: {str(e)}"


def reply_to_email(email_id: str, reply_text: str):
    """
    Reply to an existing email message.
    
    Fetches the original email, extracts sender and subject information,
    and sends a reply with the appropriate "Re:" prefix and reply-to headers.
    
    Args:
        email_id (str): The IMAP message ID of the email to reply to
        reply_text (str): The text content of the reply message
        
    Returns:
        str: Success message if reply sent, or error message if failed
        
    Example:
        >>> reply_to_email("12345", "Thanks for your message!")
        "✅ Email sent successfully to sender@example.com!"
    """
    try:
        # First, get the original email
        mail = connect_imap()
        mail.select('INBOX')
        _, msg_data = mail.fetch(email_id.encode(), '(RFC822)')
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        original_subject = decode_header(email_message['Subject'])[0][0]
        if isinstance(original_subject, bytes):
            original_subject = original_subject.decode()
        
        reply_to = email_message.get('Reply-To') or email_message.get('From')
        
        mail.close()
        mail.logout()
        
        # Create reply
        subject = f"Re: {original_subject}" if not original_subject.startswith('Re:') else original_subject
        return send_email(reply_to, subject, reply_text)
    except Exception as e:
        return f"❌ Error replying: {str(e)}"


# ==================== CALENDAR FUNCTIONS ====================

def create_event(summary: str, start_time: str, end_time: str, description: str = "", location: str = "", attendees: str = ""):
    """
    Create a new event in Google Calendar.
    
    Creates a calendar event with the specified details. Supports adding event
    description, location, and inviting attendees.
    
    Args:
        summary (str): Title/name of the event
        start_time (str): Event start time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
        end_time (str): Event end time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
        description (str, optional): Detailed event description. Defaults to ""
        location (str, optional): Event location/venue. Defaults to ""
        attendees (str, optional): Comma-separated email addresses of attendees. Defaults to ""
        
    Returns:
        str: Success message with event link or error message
        
    Example:
        >>> create_event("Team Meeting", "2025-12-01T10:00:00", "2025-12-01T11:00:00",
        ...              location="Conference Room A", attendees="user@example.com")
        "✅ Event created! Link: https://calendar.google.com/..."
    """
    try:
        creds = get_calendar_credentials()
        service = build('calendar', 'v3', credentials=creds)
        
        event = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        
        if location:
            event['location'] = location
        
        if attendees:
            event['attendees'] = [{'email': email.strip()} for email in attendees.split(',')]
        
        result = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Event created! Link: {result.get('htmlLink')}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error creating event: {str(e)}\n\nDetails:\n{error_details}"


def list_events(max_results: int = 10, days_ahead: int = 30):
    """
    Retrieve upcoming events from Google Calendar.
    
    Fetches calendar events starting from the current time and extending forward
    for the specified number of days. Returns event details including time, title,
    location, and description.
    
    Args:
        max_results (int, optional): Maximum number of events to retrieve. Defaults to 10
        days_ahead (int, optional): Number of days forward to search. Defaults to 30
        
    Returns:
        str: JSON string containing list of events or "No upcoming events." message
        Each event contains: id, summary, start, end, description, location
        
    Example:
        >>> list_events(max_results=5, days_ahead=7)
        '[{"id": "abc123", "summary": "Meeting", "start": "2025-12-01T10:00:00Z", ...}]'
    """
    try:
        creds = get_calendar_credentials()
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        time_max = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat().replace('+00:00', 'Z')
        
        results = service.events().list(
            calendarId='primary',
            timeMin=now,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = results.get('items', [])
        if not events:
            return "No upcoming events."
        
        event_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_list.append({
                'id': event['id'],
                'summary': event.get('summary', 'No Title'),
                'start': start,
                'end': event['end'].get('dateTime', event['end'].get('date')),
                'description': event.get('description', ''),
                'location': event.get('location', '')
            })
        
        return json.dumps(event_list, indent=2)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error listing events: {str(e)}\n\nDetails:\n{error_details}"


def delete_event(event_id: str):
    """
    Delete a calendar event by its ID.
    
    Permanently removes an event from the primary Google Calendar.
    
    Args:
        event_id (str): The unique identifier of the calendar event to delete
        
    Returns:
        str: Success message confirming deletion or error message
        
    Example:
        >>> delete_event("abc123xyz")
        "✅ Event abc123xyz deleted!"
    """
    try:
        creds = get_calendar_credentials()
        service = build('calendar', 'v3', credentials=creds)
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return f"✅ Event {event_id} deleted!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ==================== TOOL GETTERS ====================


def get_all_gmail_tools() -> List[StructuredTool]:
    """
    Get all Gmail tools as LangChain StructuredTools.

    Returns:
        List[StructuredTool]: List of all Gmail tools ready to be added to the tool list
    """
    return [
        StructuredTool.from_function(send_email),
        StructuredTool.from_function(list_emails),
        StructuredTool.from_function(read_email),
        StructuredTool.from_function(search_emails),
        StructuredTool.from_function(reply_to_email),
    ]


def get_all_calendar_tools() -> List[StructuredTool]:
    """
    Get all Google Calendar tools as LangChain StructuredTools.

    Returns:
        List[StructuredTool]: List of all calendar tools ready to be added to the tool list
    """
    return [
        StructuredTool.from_function(create_event),
        StructuredTool.from_function(list_events),
        StructuredTool.from_function(delete_event),
    ]


def get_all_tools() -> List[StructuredTool]:
    """
    Get all available tools (Gmail and Calendar) as LangChain StructuredTools.

    Returns:
        List[StructuredTool]: List of all tools ready to be added to the tool list
    """
    return (
        get_all_gmail_tools() +
        get_all_calendar_tools()
    )


if __name__ == "__main__":
    """
    Test the custom tools independently.
    You can modify the test cases below to test different functionalities.
    """
    print("=" * 80)
    print("Testing Custom Tools - Gmail and Calendar")
    print("=" * 80)
    
    # Test getting all tools
    print("\n📋 Loading all tools...")
    all_tools = get_all_tools()
    print(f"✅ Loaded {len(all_tools)} tools:")
    for i, tool in enumerate(all_tools, 1):
        print(f"  {i}. {tool.name}: {tool.description}")
    
    print("\n" + "=" * 80)
    print("Gmail Tools Test")
    print("=" * 80)
    
    # Test Gmail tools
    print("\n📧 Testing list_emails (last 3 emails)...")
    try:
        result = list_emails(limit=3)
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Calendar Tools Test")
    print("=" * 80)
    
    # Test Calendar tools
    print("\n📅 Testing list_events (next 5 events)...")
    try:
        result = list_events(max_results=5, days_ahead=30)
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


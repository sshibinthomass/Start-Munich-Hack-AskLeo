# BotBuild - AI Negotiation & Communication Assistant

## 📋 Table of Contents
- [What is BotBuild?](#what-is-botbuild)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [Troubleshooting](#troubleshooting)

---

## 🎯 What is BotBuild?

**BotBuild** is an intelligent AI negotiation assistant built for the Start Munich Hackathon. It automates business negotiations with vendors, manages communications via email and calendar, and provides a voice-enabled interface.

### What It Does

1. **Automated Negotiation**: Negotiates with vendors using strategic conversation tactics
2. **Multi-Agent Communication**: Two AI agents can negotiate with each other automatically
3. **Email & Calendar Integration**: Sends confirmation emails and schedules meetings after deals
4. **Voice Interface**: Talk to the AI and hear responses with natural voices
5. **Product Knowledge**: Knows about Victoria Arduino espresso machines

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│         React Frontend (Port 5173)                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  • Chat Interface                            │  │
│  │  • Voice Recording (Microphone)              │  │
│  │  • Agent-to-Agent Controls                   │  │
│  │  • Product Landing Page                      │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                      │ HTTP/SSE
                      ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI Backend (Port 8000)                 │
│  ┌──────────────────────────────────────────────┐  │
│  │  Main Routes:                                │  │
│  │  • /chat - Chat with AI                      │  │
│  │  • /chat/agent-to-agent - Auto negotiation   │  │
│  │  • /tts - Text-to-speech                     │  │
│  │  • /transcribe - Speech-to-text              │  │
│  │  • /products - Product catalog               │  │
│  └──────────────────────────────────────────────┘  │
│                      │                              │
│  ┌──────────────────┴──────────────────────────┐   │
│  │   LangGraph Agent (MCPChatbotNode)          │   │
│  │   • Message processing                       │   │
│  │   • Tool execution                           │   │
│  │   • ScoutBot negotiation logic               │   │
│  └──────────────────────────────────────────────┘   │
│                      │                              │
│  ┌──────────┬────────┴──────┬──────────────────┐   │
│  │  Tools   │  LLM Provider │  External APIs   │   │
│  │          │               │                  │   │
│  │ • Gmail  │  • OpenAI     │  • Dunkler API   │   │
│  │ • Calendar│   - GPT-4o   │  • ElevenLabs    │   │
│  │ • Product │   - Whisper  │    (TTS)         │   │
│  │   Catalog │              │                  │   │
│  └──────────┴───────────────┴──────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 AI Agent
- Strategic buyer negotiation bot ("ScoutBot")
- Keeps responses concise and professional
- Automatically detects when a deal is reached
- Sends emails and schedules meetings after successful deals

### 📧 Communication
- **Gmail**: Send, read, search, and reply to emails
- **Google Calendar**: Create, list, and delete events

### 🎙️ Voice
- **Speech-to-Text**: Speak your messages using OpenAI Whisper
- **Text-to-Speech**: Hear AI responses with ElevenLabs voices

### 🏢 Product Catalog
- Victoria Arduino espresso machines:
  - Maverick Gravimetric (Flagship) - $32,900
  - Eagle Tempo Digit (Mid-Range) - $19,900
  - E1 Prima Volumetric (Entry-Level) - $7,490

---

## 📋 Prerequisites

### Required Software
- **Python 3.13+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **Git** - [Download](https://git-scm.com/)

### Required API Keys

1. **OpenAI API Key** (REQUIRED)
   - For: GPT models and Whisper speech-to-text
   - Get it: https://platform.openai.com/api-keys

2. **ElevenLabs API Key** (REQUIRED)
   - For: Text-to-speech
   - Get it: https://elevenlabs.io/

3. **Gmail App Password** (REQUIRED)
   - For: Email features
   - Setup: Google Account → Security → 2-Step Verification → App passwords

4. **Google OAuth Credentials** (REQUIRED)
   - For: Google Calendar
   - Setup: https://console.cloud.google.com/
   - Create OAuth 2.0 Client ID (Desktop Application)

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/sshibinthomass/Start-Munich-Hack-AskLeo.git
cd Start-Munich-Hack-AskLeo
```

### Step 2: Backend Setup

```bash
cd backend

# Install dependencies
pip install uv
uv sync

# Or use pip
pip install -e .
```

### Step 3: Configure Environment

```bash
# Create .env file
cp example.env .env

# Edit .env with your API keys
notepad .env  # Windows
# nano .env   # Mac/Linux
```

**Fill in your `.env` file:**

```bash
# OpenAI (REQUIRED)
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE

# ElevenLabs (REQUIRED)
ELEVENLABS_API_KEY=sk_YOUR_KEY_HERE

# Gmail (REQUIRED)
USER_GOOGLE_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Google Calendar OAuth (REQUIRED)
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=GOCSPX-your-secret

# MCP Server Paths (Update with your actual paths)
MCP_FOLDER_DIR=C:\Users\YourUsername\Start-Munich-Hack-AskLeo\backend\langgraph_agent\mcps\local_servers\dataflow.py
MCP_FILESYSTEM_DIR=C:\Users\YourUsername\Start-Munich-Hack-AskLeo\backend\langgraph_agent\mcps\local_servers\dataflow.py
MCP_PRODUCT_DIR=C:\Users\YourUsername\Start-Munich-Hack-AskLeo\backend\langgraph_agent\mcps\local_servers\products.py
```

### Step 4: Setup Google Calendar OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **Google Calendar API**
4. Create **OAuth 2.0 Client ID** (choose "Desktop app")
5. Download the JSON file
6. Save it as `backend/client_secret_<client-id>.apps.googleusercontent.com.json`

### Step 5: Frontend Setup

```bash
cd ../react_frontend
npm install
```

---

## 🏃 Running the Application

### Open Two Terminals

**Terminal 1 - Backend:**
```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd react_frontend
npm run dev
```

### Access the Application

Open your browser: **http://localhost:5173**

---

## 🎯 How to Use

### Basic Chat

1. Select "OpenAI" provider and a model (e.g., "GPT-4o Mini")
2. Type your message
3. Press Enter or click Send
4. Watch the AI respond in real-time

**Example:**
```
You: "What Victoria Arduino machines are available?"
BotBuild: "We have three models: Maverick ($32,900), 
          Eagle Tempo ($19,900), and E1 Prima ($7,490)."
```

### Voice Chat

1. Click the **microphone icon**
2. Speak your message
3. Click stop when done
4. Message appears automatically
5. Enable "Voice Output" to hear responses

### Agent-to-Agent Negotiation

1. Click **"Agent-to-Agent"** tab
2. Choose conversation mode:
   - **Fixed**: Runs for exact number of exchanges (default: 11)
   - **Until Deal**: Runs until deal detected (max: 15)
3. Click **"Start Agent Negotiation"**
4. Watch BotBuild and Dunkler negotiate automatically

**When deal is reached:**
- ✅ Confirmation email sent to vendor
- ✅ Meeting scheduled (2 days later at 10 AM)

### Using Tools

The AI can automatically:
- **Send emails** - "Send an email to vendor@example.com"
- **Check emails** - "Do I have any new emails?"
- **Schedule meetings** - "Schedule a meeting tomorrow at 2 PM"
- **Get product info** - "Tell me about the Maverick machine"

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
cd backend
uv sync
```

### Gmail Not Working

**Error**: `Authentication failed`

**Solution**:
- Use Gmail App Password (NOT regular password)
- Generate at: Google Account → Security → App passwords
- Format: `GMAIL_APP_PASSWORD=xxxxnxxxnxxxnxxxx` (no spaces)

### Calendar Authentication Keeps Opening Browser

**Solution**:
1. Delete `calendar_token.json` if it exists
2. Make sure OAuth client is "Desktop App" type
3. Complete OAuth flow once
4. Token will be saved automatically

### Voice Not Working

**Solution**:
- **TTS**: Check `ELEVENLABS_API_KEY` is valid
- **STT**: Check `OPENAI_API_KEY` is valid
- Check browser microphone permissions

### MCP Tools Not Loading

**Solution**:
- Use **absolute paths** in `.env` for MCP_*_DIR variables
- Example (Windows): `C:\Users\...\backend\langgraph_agent\mcps\local_servers\products.py`
- Example (Mac/Linux): `/Users/.../backend/langgraph_agent/mcps/local_servers/products.py`

---

## 📊 Product Catalog

### Victoria Arduino Espresso Machines

**1. Maverick Gravimetric 3gr**
- Price: $32,900 (negotiable to $29,610)
- 3 groups, gravimetric extraction
- 5 units in stock, 5-day delivery

**2. Eagle Tempo Digit 3gr**
- Price: $19,900 (negotiable to $17,900)
- 3 groups, digital dosing
- 12 units in stock, 3-day delivery

**3. E1 Prima Volumetric T3**
- Price: $7,490 (negotiable to $6,990)
- 1 group, compact design
- 18 units in stock, 2-day delivery

---

## 📝 Example Usage

### Negotiation Example
```
You: "I need 20 Maverick machines. Can you offer a discount?"
BotBuild: "For 20 units at $32,900 each, I can offer 8% off. 
          That's $30,268 per unit. Deal?"
You: "Deal! Send me a confirmation email."
[Tool executes: send_email]
BotBuild: "Confirmation sent! Meeting scheduled for 2 days from now."
```

### Email Example
```
You: "Check my emails"
[Tool executes: list_emails]
BotBuild: "You have 3 new emails: 
          1. Order confirmation from supplier
          2. Invoice from vendor..."
```

---

## 📞 Support

- **GitHub**: [Start-Munich-Hack-AskLeo](https://github.com/sshibinthomass/Start-Munich-Hack-AskLeo)
- **Issues**: Open an issue on GitHub

---

*Built for Start Munich Hackathon 2025*

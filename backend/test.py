import requests

# Configuration
API_BASE = "https://negbot-backend-ajdxh9axb0ddb0e9.westeurope-01.azurewebsites.net/api"
TEAM_ID = 851996
VENDOR_ID = 44

print("🤝 Negotiation Bot - Vendor 44 Chat")
print("=" * 50)

# Create a new conversation
print(f"\n📝 Creating conversation with Vendor {VENDOR_ID}...")
response = requests.post(
    f"{API_BASE}/conversations/",
    params={"team_id": TEAM_ID},
    json={"vendor_id": VENDOR_ID, "title": "Price Negotiation Session"},
)

if response.status_code not in [200, 201]:
    print(f"❌ Failed to create conversation: {response.status_code}")
    print(response.text)
    exit(1)

conversation_id = response.json()["id"]
print(f"✅ Conversation created (ID: {conversation_id})")

# Chat loop
print("\n💬 Start chatting! Type 'quit' or 'exit' to stop.\n")

while True:
    user_message = input("You: ").strip()

    if not user_message:
        continue

    if user_message.lower() in ["quit", "exit", "stop"]:
        print("\n👋 Goodbye!")
        break

    # Send message and get AI response
    response = requests.post(
        f"{API_BASE}/messages/{conversation_id}", data={"content": user_message}
    )

    if response.status_code in [200, 201]:
        ai_response = response.json()
        print(f"\n🤖 Vendor {VENDOR_ID}: {ai_response.get('content', 'No response')}\n")
    else:
        print(f"❌ Error: {response.status_code}")
        if response.status_code == 429:
            print("⏳ Rate limit reached. Wait a moment and try again.")

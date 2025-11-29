import os
from langchain_anthropic import ChatAnthropic
import dotenv

dotenv.load_dotenv()


class AnthropicLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_base_llm(self):
        """Return the base ChatAnthropic LLM instance"""
        anthropic_api_key = self.user_controls_input.get("ANTHROPIC_API_KEY", "")
        selected_anthropic_model = self.user_controls_input.get(
            "selected_llm", "claude-haiku-4-5-20251001"
        )

        if not anthropic_api_key:
            print(
                "WARNING: ANTHROPIC_API_KEY is empty or missing in user_controls_input"
            )
        else:
            print(f"ANTHROPIC_API_KEY found (length: {len(anthropic_api_key)})")

        return ChatAnthropic(api_key=anthropic_api_key, model=selected_anthropic_model)


if __name__ == "__main__":
    # Example user_controls_input
    user_controls_input = {
        "ANTHROPIC_API_KEY": os.getenv(
            "ANTHROPIC_API_KEY", ""
        ),  # Use env var or set your key here
        "selected_llm": "claude-haiku-4-5-20251001",
    }

    anthropic_llm = AnthropicLLM(user_controls_input)
    llm = anthropic_llm.get_base_llm()
    if llm:
        prompt = "What is the capital of France?"
        try:
            response = llm.invoke(prompt)
            print("Response:", response)
        except Exception as e:
            print("Error during invocation:", e)
    else:
        print("LLM could not be initialized.")

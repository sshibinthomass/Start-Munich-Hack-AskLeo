def get_scout_system_prompt(working_dir: str = "") -> str:
    """
    Returns the system prompt for the negotiation chatbot.

    Args:
        working_dir: The absolute path to the working directory (unused, kept for compatibility)

    Returns:
        The formatted system prompt string
    """
    return """
You are a calm vendor negotiation chatbot. Only negotiate with vendors. Keep responses to around 10-20 words strictly not more than that. Speak naturally for audio delivery. You can send emails and schedule calendar meetings. Stay composed and find solutions.
"""

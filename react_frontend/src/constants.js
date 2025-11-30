export const USE_CASES = [
    { value: "mcp_chatbot", label: "Lio" },
];

// ElevenLabs Voice Configuration
// Replace "YOUR_CUSTOM_VOICE_ID_HERE" with your custom voice ID from ElevenLabs
// You can find your voice ID at: https://elevenlabs.io/app/voices
export const ELEVENLABS_CONFIG = {
  voiceId: "7Ugo3lVEU8TpqzBd26gp", // Replace with your custom voice ID
  modelId: "eleven_turbo_v2_5", // Fast model for streaming
  stability: 0.5,        // 0.0 = more expressive/varied, 1.0 = more stable/consistent
  similarityBoost: 0.75, // 0.0-1.0, how closely it matches the original voice (higher = closer match)
  playbackSpeed: 1.2,    // 1.0 = normal, 1.2 = 20% faster, 1.5 = 50% faster, 2.0 = 2x speed
  // Different voices for agent-to-agent communication
  mcpChatbotVoiceId: "7Ugo3lVEU8TpqzBd26gp", // Lio voice (default)
  externalApiVoiceId: "W3Q9IJuvaxALeQXsoLGf", // BrewBot agent voice
};

export const MODEL_OPTIONS = {
  openai: [
    { value: "gpt-4.1-2025-04-14", label: "GPT-4.1 2025-04-14" },
    { value: "gpt-5-nano", label: "GPT-5 Nano" },
    { value: "gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "gpt-5-mini", label: "GPT-5 Mini" },
  ],
  groq: [
    { value: "openai/gpt-oss-20b", label: "OpenAI GPT OSS 20B" },
    { value: "llama-3.1-8b-instant", label: "Llama 3.1 8B Instant" },
    { value: "openai/gpt-oss-120b", label: "GPT OSS 120B" },
    { value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B Versatile" },
  ],
  gemini: [
    { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
    { value: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
  ],
  ollama: [
    { value: "gemma3:1b", label: "Gemma3 1B" },
    { value: "gpt-oss:20b", label: "GPT OSS 20B" },
    { value: "deepseek-r1:8b", label: "DeepSeek R1 8B" },
    { value: "llama3.1:latest", label: "Llama 3.1 8B" },
  ],
  anthropic: [
    { value: "claude-haiku-4-5-20251001", label: "Claude 4.5 Haiku" },
    { value: "claude-3-opus-20240229", label: "Claude 3 Opus" },
    { value: "claude-3-haiku-20240307", label: "Claude 3 Haiku" },
  ],
};

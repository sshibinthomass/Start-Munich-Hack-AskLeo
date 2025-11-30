import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { MODEL_OPTIONS, USE_CASES, ELEVENLABS_CONFIG } from "./constants";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { ProductLanding } from "./components/ProductLanding";
import { FloatingChat } from "./components/FloatingChat";
import { escapeHtml, renderMarkdown } from "./utils/markdown";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function App() {
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resetting, setResetting] = useState(false);
  const [sessionId, setSessionId] = useState("default");
  const [provider] = useState("openai");
  const [model, setModel] = useState(() => MODEL_OPTIONS.openai[0].value);
  const [useCase, setUseCase] = useState("mcp_chatbot");
  const [backendStatus, setBackendStatus] = useState("checking");
  const [backendStatusMessage, setBackendStatusMessage] = useState("Checking backend...");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const toolStatusEventSourceRef = useRef(null);
  const [toolStatusStreamKey, setToolStatusStreamKey] = useState(0);
  const [latestToolCall, setLatestToolCall] = useState(null);
  const [toolCallHistory, setToolCallHistory] = useState([]);
  const [toolStatusComplete, setToolStatusComplete] = useState(true);
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const audioQueueRef = useRef([]);
  const currentAudioRef = useRef(null);
  const textBufferRef = useRef("");
  const isPlayingRef = useRef(false);
  const [agentToAgentEnabled, setAgentToAgentEnabled] = useState(false);
  const [agentToAgentLoading, setAgentToAgentLoading] = useState(false);
  const [dunklerConversationId, setDunklerConversationId] = useState(null);
  const [maxExchanges, setMaxExchanges] = useState(11);
  const [conversationMode, setConversationMode] = useState("fixed"); // "fixed" or "until_deal"
  const [initialMessage, setInitialMessage] = useState("hello");

  const normalizeToolEntry = (entry) => {
    if (!entry) return null;
    if (typeof entry === "string") {
      return {
        id: null,
        name: entry,
        timestamp: "",
        args: {},
        response: "",
        duration_ms: null,
      };
    }
    if (typeof entry !== "object") return null;
    const name =
      typeof entry.name === "string"
        ? entry.name
        : entry.name !== undefined
        ? String(entry.name)
        : "Unknown tool";
    const timestamp = typeof entry.timestamp === "string" ? entry.timestamp : "";
    const args =
      entry.args && typeof entry.args === "object"
        ? entry.args
        : {};
    const response =
      typeof entry.response === "string"
        ? entry.response
        : entry.response !== undefined
        ? JSON.stringify(entry.response)
        : "";
    const duration =
      typeof entry.duration_ms === "number" ? entry.duration_ms : null;
    const id =
      typeof entry.id === "string"
        ? entry.id
        : entry.id !== undefined
        ? String(entry.id)
        : null;
    return {
      id,
      name,
      timestamp,
      args,
      response,
      duration_ms: duration,
    };
  };

  const normalizeToolEntries = (value) => {
    if (!Array.isArray(value)) return [];
    return value
      .map((entry) => normalizeToolEntry(entry))
      .filter((entry) => entry && entry.name);
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const existing = localStorage.getItem("chat_session_id");
    if (existing) {
      setSessionId(existing);
    } else {
      const id =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `session-${Date.now()}`;
      localStorage.setItem("chat_session_id", id);
      setSessionId(id);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`, {
          method: "GET",
        });

        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }

        const data = await response.json().catch(() => ({}));

        if (!isMounted) return;
        setBackendStatus("online");
        setBackendStatusMessage(
          typeof data === "object" && data !== null
            ? data.status ?? "Online"
            : "Online"
        );
      } catch (err) {
        if (!isMounted) return;
        setBackendStatus("offline");
        setBackendStatusMessage(
          err instanceof Error ? err.message : "Unable to reach backend"
        );
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 15000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleModelChange = (nextModel) => {
    setModel(nextModel);
  };

  useEffect(() => {
    if (!sessionId) return;
    const params = new URLSearchParams({
      session_id: sessionId || "default",
      use_case: useCase,
    });
    const eventSource = new EventSource(`${BACKEND_URL}/tool-status/stream?${params.toString()}`);
    toolStatusEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLatestToolCall(normalizeToolEntry(data?.latest_tool_call));
        setToolCallHistory(normalizeToolEntries(data?.tool_calls));
        setToolStatusComplete(Boolean(data?.completed));
      } catch (err) {
        // Ignore malformed events
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      if (toolStatusEventSourceRef.current === eventSource) {
        toolStatusEventSourceRef.current = null;
      }
      setTimeout(() => setToolStatusStreamKey((prev) => prev + 1), 1500);
    };

    return () => {
      eventSource.close();
      if (toolStatusEventSourceRef.current === eventSource) {
        toolStatusEventSourceRef.current = null;
      }
    };
  }, [sessionId, useCase, BACKEND_URL, toolStatusStreamKey]);

  const resetConversation = async () => {
    setError("");
    setResetting(true);
    setLatestToolCall(null);
    setToolCallHistory([]);
    setToolStatusComplete(true);
    setDunklerConversationId(null); // Clear Dunkler conversation ID
    try {
      const res = await fetch(`${BACKEND_URL}/chat/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId || "default",
          use_case: useCase,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Unable to clear conversation.");
      }
      setConversation([]);
    } catch (err) {
      setError(err.message || "Failed to clear conversation.");
    } finally {
      setResetting(false);
    }
  };


  // Function to process text chunks for TTS
  const processTextForTTS = (newText) => {
    if (!voiceOutputEnabled) return;
    
    textBufferRef.current += newText;
    
    // Split by sentence endings (., !, ?, followed by space or end)
    const sentenceEndRegex = /([.!?])\s+/g;
    const sentences = [];
    let lastIndex = 0;
    let match;
    
    while ((match = sentenceEndRegex.exec(textBufferRef.current)) !== null) {
      const sentence = textBufferRef.current.substring(lastIndex, match.index + 1).trim();
      if (sentence.length > 0) {
        sentences.push(sentence);
      }
      lastIndex = match.index + match[0].length;
    }
    
    // If we found complete sentences, process them
    if (sentences.length > 0) {
      // Update buffer to keep remaining text
      textBufferRef.current = textBufferRef.current.substring(lastIndex);
      
      // Queue each sentence for TTS
      sentences.forEach((sentence) => {
        if (sentence.length > 0) {
          queueAudioForText(sentence);
        }
      });
    }
  };

  // Function to queue audio for text (for regular chat)
  const queueAudioForText = async (text) => {
    if (!voiceOutputEnabled) return;
    
    try {
      const response = await fetch(`${BACKEND_URL}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          voice_id: ELEVENLABS_CONFIG.voiceId,
          model_id: ELEVENLABS_CONFIG.modelId,
          stability: ELEVENLABS_CONFIG.stability,
          similarity_boost: ELEVENLABS_CONFIG.similarityBoost,
        }),
      });

      if (!response.ok) {
        console.error("TTS request failed:", response.status);
        return;
      }

      // Convert response to blob and create audio URL
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Add to queue (as string for backward compatibility with regular chat)
      audioQueueRef.current.push(audioUrl);
      
      // Start playing if not already playing
      if (!isPlayingRef.current) {
        playNextAudio();
      }
    } catch (err) {
      console.error("Error generating TTS:", err);
    }
  };

  // Function to stop audio playback
  const stopAudio = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    
    // Clean up queued audio URLs
    audioQueueRef.current.forEach((item) => {
      const url = typeof item === 'string' ? item : item.audioUrl;
      URL.revokeObjectURL(url);
      // Resolve any pending promises
      if (typeof item === 'object' && item.resolve) {
        item.resolve();
      }
    });
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsAudioPlaying(false);
  };

  // Function to play next audio in queue
  const playNextAudio = () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      setIsAudioPlaying(false);
      return;
    }

    isPlayingRef.current = true;
    setIsAudioPlaying(true);
    const queueItem = audioQueueRef.current.shift();
    const audioUrl = typeof queueItem === 'string' ? queueItem : queueItem.audioUrl;
    const resolveCallback = typeof queueItem === 'object' && queueItem.resolve ? queueItem.resolve : null;
    
    const audio = new Audio(audioUrl);
    currentAudioRef.current = audio;
    
    // Set playback speed (1.0 = normal, 1.2 = 20% faster, etc.)
    audio.playbackRate = ELEVENLABS_CONFIG.playbackSpeed || 1.0;
    
    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
      if (resolveCallback) {
        resolveCallback();
      }
      playNextAudio(); // Play next in queue
    };
    
    audio.onerror = () => {
      console.error("Audio playback error");
      URL.revokeObjectURL(audioUrl);
      if (resolveCallback) {
        resolveCallback();
      }
      isPlayingRef.current = false;
      setIsAudioPlaying(false);
      playNextAudio(); // Try next
    };
    
    audio.onpause = () => {
      // If audio is paused (not ended), update state
      if (audioQueueRef.current.length === 0 && !audio.ended) {
        isPlayingRef.current = false;
        setIsAudioPlaying(false);
      }
    };
    
    audio.play().catch((err) => {
      console.error("Error playing audio:", err);
      URL.revokeObjectURL(audioUrl);
      if (resolveCallback) {
        resolveCallback();
      }
      isPlayingRef.current = false;
      setIsAudioPlaying(false);
      playNextAudio();
    });
  };

  // Cleanup audio on unmount or when voice output is disabled
  useEffect(() => {
    return () => {
      // Stop current audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      
      // Clean up queued audio URLs
      audioQueueRef.current.forEach((item) => {
        const url = typeof item === 'string' ? item : item.audioUrl;
        URL.revokeObjectURL(url);
        // Resolve any pending promises
        if (typeof item === 'object' && item.resolve) {
          item.resolve();
        }
      });
      audioQueueRef.current = [];
      isPlayingRef.current = false;
      textBufferRef.current = "";
    };
  }, []);

  // Stop audio when voice output is disabled
  useEffect(() => {
    if (!voiceOutputEnabled) {
      stopAudio();
      textBufferRef.current = "";
    }
  }, [voiceOutputEnabled]);

  // Handler for voice output toggle button
  const handleVoiceOutputToggle = () => {
    // If audio is playing, stop it instead of toggling voice output
    if (isAudioPlaying) {
      stopAudio();
    } else {
      // Otherwise, toggle voice output setting
      setVoiceOutputEnabled(!voiceOutputEnabled);
    }
  };

  // Function to queue audio for agent message with specific voice
  const queueAudioForAgent = async (text, agent) => {
    if (!voiceOutputEnabled) return Promise.resolve();
    
    try {
      // Use different voice for each agent
      const voiceId = agent === "mcp_chatbot" 
        ? ELEVENLABS_CONFIG.mcpChatbotVoiceId 
        : ELEVENLABS_CONFIG.externalApiVoiceId;
      
      const response = await fetch(`${BACKEND_URL}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          voice_id: voiceId,
          model_id: ELEVENLABS_CONFIG.modelId,
          stability: ELEVENLABS_CONFIG.stability,
          similarity_boost: ELEVENLABS_CONFIG.similarityBoost,
        }),
      });

      if (!response.ok) {
        console.error("TTS request failed:", response.status);
        return Promise.resolve();
      }

      // Convert response to blob and create audio URL
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Create a promise that resolves when this specific audio finishes
      return new Promise((resolve) => {
        // Add to queue with resolve callback
        audioQueueRef.current.push({ audioUrl, resolve });
        
        // Start playing if not already playing
        if (!isPlayingRef.current) {
          playNextAudio();
        }
      });
    } catch (err) {
      console.error("Error generating TTS:", err);
      return Promise.resolve();
    }
  };

  // Handler for agent-to-agent toggle
  const handleAgentToAgentToggle = async (customInitialMessage = null) => {
    if (agentToAgentEnabled) {
      // Disable agent-to-agent mode (but keep conversation ID for continued chat)
      setAgentToAgentEnabled(false);
      setAgentToAgentLoading(false);
      stopAudio(); // Stop any playing audio
      // Don't clear dunklerConversationId - user can continue chatting
      return;
    }

    // Get the initial message from the input or use the provided one
    // We'll get this from the FloatingChat component via a callback
    const messageToUse = customInitialMessage || initialMessage || "hello";

    // Enable agent-to-agent mode and start conversation
    setAgentToAgentEnabled(true);
    setAgentToAgentLoading(true);
    setError("");
    
    // Clear existing conversation and audio
    setConversation([]);
    setLatestToolCall(null);
    setToolCallHistory([]);
    setToolStatusComplete(true);
    setDunklerConversationId(null); // Reset conversation ID
    stopAudio();

    try {
      const response = await fetch(`${BACKEND_URL}/chat/agent-to-agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          selected_llm: model,
          max_exchanges: conversationMode === "fixed" ? maxExchanges : null,
          conversation_mode: conversationMode,
          initial_message: messageToUse,
          voice_output_enabled: voiceOutputEnabled,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Agent-to-agent backend returned an error.");
      }

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "status") {
                // Status message (e.g., "Dunkler agent initialized")
                setConversation((prev) => [
                  ...prev,
                  {
                    text: data.message,
                    rendered: escapeHtml(data.message),
                    isUser: false,
                    agent: "system",
                  },
                ]);
              } else if (data.type === "agent_message") {
                // Message from an agent
                const agentName = data.agent === "mcp_chatbot" ? "Leo" : "Dunkler";
                const messageText = `[${agentName}]: ${data.message}`;
                const rendered = renderMarkdown(data.message);
                
                setConversation((prev) => [
                  ...prev,
                  {
                    text: data.message,
                    rendered: `<strong>${agentName}:</strong> ${rendered}`,
                    isUser: false,
                    agent: data.agent,
                  },
                ]);

                // Queue audio for this message if voice output is enabled and wait for it to complete
                if (voiceOutputEnabled) {
                  await queueAudioForAgent(data.message, data.agent);
                }
              } else if (data.type === "error") {
                const errorMsg = data.error || "Unknown error occurred";
                setError(`Error from ${data.agent || "system"}: ${errorMsg}`);
                setAgentToAgentLoading(false);
              } else if (data.type === "done") {
                const completionMessage = data.deal_reached
                  ? `Deal reached after ${data.exchanges || 0} exchanges! You can now continue chatting with Dunkler.`
                  : `Conversation completed after ${data.exchanges || 0} exchanges. You can now continue chatting with Dunkler.`;
                
                setConversation((prev) => [
                  ...prev,
                  {
                    text: completionMessage,
                    rendered: escapeHtml(completionMessage),
                    isUser: false,
                    agent: "system",
                  },
                ]);
                setAgentToAgentLoading(false);
                // Store conversation ID for continued chat with Dunkler
                if (data.conversation_id) {
                  setDunklerConversationId(data.conversation_id);
                }
              }
            } catch (parseError) {
              console.error("Error parsing SSE data:", parseError);
            }
          }
        }
      }
    } catch (err) {
      let message = err.message || "Something went wrong";
      setError(message);
      setAgentToAgentEnabled(false);
      setAgentToAgentLoading(false);
    }
  };

  async function handleSubmitMessage(content) {
    setError("");

    const trimmed = content.trim();
    if (!trimmed) return;

    // Check if we should send to Dunkler (after agent-to-agent conversation)
    // Priority: If Dunkler conversation exists, always send to Dunkler (not Leo)
    if (dunklerConversationId) {
      // Send message to Dunkler
      const userMessage = {
        text: trimmed,
        rendered: escapeHtml(trimmed),
        isUser: true,
      };
      setConversation((prev) => [...prev, userMessage]);
      setLoading(true);

      // Add empty bot message that we'll update
      setConversation((prev) => [
        ...prev,
        { text: "", rendered: "", isUser: false, agent: "external_api" },
      ]);

      try {
        const response = await fetch(`${BACKEND_URL}/chat/dunkler`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: trimmed,
            conversation_id: dunklerConversationId,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || "Dunkler backend returned an error.");
        }

        const data = await response.json();
        const dunklerResponse = data.response || "No response";
        const rendered = renderMarkdown(dunklerResponse);

        setConversation((prev) => {
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          if (lastIndex >= 0 && !updated[lastIndex].isUser) {
            updated[lastIndex] = {
              text: dunklerResponse,
              rendered: `<strong>Dunkler:</strong> ${rendered}`,
              isUser: false,
              agent: "external_api",
            };
          }
          return updated;
        });

        // Queue audio if voice output is enabled
        if (voiceOutputEnabled) {
          await queueAudioForAgent(dunklerResponse, "external_api");
        }

        setLoading(false);
        return;
      } catch (err) {
        let message = err.message || "Something went wrong";
        setError(message);
        setLoading(false);
        setConversation((prev) => prev.slice(0, -1)); // Remove empty bot message
        return;
      }
    }

    // Normal chat with Leo
    const userMessage = {
      text: trimmed,
      rendered: escapeHtml(trimmed),
      isUser: true,
    };
    setConversation((prev) => [...prev, userMessage]);
    setLoading(true);
    setLatestToolCall(null);
    setToolCallHistory([]);
    setToolStatusComplete(false);
    
    // Reset text buffer for new response
    textBufferRef.current = "";

    // Add empty bot message that we'll update as chunks arrive
    setConversation((prev) => [
      ...prev,
      { text: "", rendered: "", isUser: false },
    ]);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          provider,
          selected_llm: model,
          use_case: useCase,
          session_id: sessionId || "default",
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Chatbot backend returned an error.");
      }

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === "chunk") {
                fullResponse += data.content;
                const rendered = renderMarkdown(fullResponse);
                setConversation((prev) => {
                  const updated = [...prev];
                  // Update the last message (which should be the bot's message)
                  const lastIndex = updated.length - 1;
                  if (lastIndex >= 0 && !updated[lastIndex].isUser) {
                    updated[lastIndex] = {
                      text: fullResponse,
                      rendered,
                      isUser: false,
                    };
                  }
                  return updated;
                });
                
                // Process chunk for TTS if voice output is enabled
                if (voiceOutputEnabled) {
                  processTextForTTS(data.content);
                }
              } else if (data.type === "done") {
                const normalizedCalls = normalizeToolEntries(data?.tool_calls);
                setToolCallHistory(normalizedCalls);
                const latest =
                  normalizeToolEntry(data?.latest_tool_call) ??
                  (normalizedCalls.length ? normalizedCalls[normalizedCalls.length - 1] : null);
                setLatestToolCall(latest);
                setToolStatusComplete(true);
                
                // Process any remaining text in buffer for TTS
                if (voiceOutputEnabled && textBufferRef.current.trim().length > 0) {
                  queueAudioForText(textBufferRef.current.trim());
                  textBufferRef.current = "";
                }
              } else if (data.type === "error") {
                throw new Error(data.error || "Unknown error occurred");
              }
            } catch (parseError) {
              console.error("Error parsing SSE data:", parseError);
            }
          }
        }
      }
    } catch (err) {
      let message = err.message || "Something went wrong";
      if (message.includes("Chatbot not initialized")) {
        message =
          "Chatbot service is not ready yet. Please ensure the backend is running with a valid API key.";
      }
      setError(message);
      setToolStatusComplete(true);
      setLatestToolCall(null);
      setToolCallHistory([]);
      // Remove the empty bot message on error
      setConversation((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  const availableModels = MODEL_OPTIONS[provider] ?? [];
  const activeUseCaseLabel =
    USE_CASES.find((option) => option.value === useCase)?.label || "Chat";

  return (
    <div className="app-landing">
      <ProductLanding />
      <FloatingChat
        conversation={conversation}
        onSubmit={handleSubmitMessage}
        onClear={resetConversation}
        loading={loading}
        resetting={resetting}
        error={error}
        useCaseLabel={activeUseCaseLabel}
        latestToolCall={latestToolCall}
        toolCallHistory={toolCallHistory}
        toolStatusComplete={toolStatusComplete}
        backendUrl={BACKEND_URL}
        voiceOutputEnabled={voiceOutputEnabled}
        onVoiceOutputToggle={handleVoiceOutputToggle}
        isAudioPlaying={isAudioPlaying}
        agentToAgentEnabled={agentToAgentEnabled}
        onAgentToAgentToggle={handleAgentToAgentToggle}
        agentToAgentLoading={agentToAgentLoading}
        maxExchanges={maxExchanges}
        onMaxExchangesChange={setMaxExchanges}
        conversationMode={conversationMode}
        onConversationModeChange={setConversationMode}
        initialMessage={initialMessage}
        onInitialMessageChange={setInitialMessage}
      />
    </div>
  );
}



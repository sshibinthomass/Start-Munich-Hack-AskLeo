import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { MODEL_OPTIONS, USE_CASES } from "./constants";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
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


  async function handleSubmitMessage(content) {
    setError("");

    const trimmed = content.trim();
    if (!trimmed) return;

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
              } else if (data.type === "done") {
                const normalizedCalls = normalizeToolEntries(data?.tool_calls);
                setToolCallHistory(normalizedCalls);
                const latest =
                  normalizeToolEntry(data?.latest_tool_call) ??
                  (normalizedCalls.length ? normalizedCalls[normalizedCalls.length - 1] : null);
                setLatestToolCall(latest);
                setToolStatusComplete(true);
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
    <div className={`app ${sidebarOpen ? "app--sidebar-open" : "app--sidebar-closed"}`}>
      {sidebarOpen && (
        <Sidebar
          model={model}
          models={availableModels}
          useCase={useCase}
          useCases={USE_CASES}
          onModelChange={handleModelChange}
          backendUrl={BACKEND_URL}
          backendStatus={backendStatus}
          backendStatusMessage={backendStatusMessage}
        />
      )}

      <ChatWindow
        conversation={conversation}
        onSubmit={handleSubmitMessage}
        onClear={resetConversation}
        loading={loading}
        resetting={resetting}
        error={error}
        useCaseLabel={activeUseCaseLabel}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        latestToolCall={latestToolCall}
        toolCallHistory={toolCallHistory}
        toolStatusComplete={toolStatusComplete}
        backendUrl={BACKEND_URL}
      />
    </div>
  );
}



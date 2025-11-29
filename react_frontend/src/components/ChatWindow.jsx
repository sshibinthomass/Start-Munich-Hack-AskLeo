import React, { useEffect, useRef, useState } from "react";

export function ChatWindow({
  conversation,
  onSubmit,
  onClear,
  loading,
  resetting,
  error,
  useCaseLabel,
  sidebarOpen,
  onToggleSidebar,
  latestToolCall = null,
  toolCallHistory = [],
  toolStatusComplete = true,
  backendUrl = "http://localhost:8000",
}) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [showToolHistory, setShowToolHistory] = useState(false);
  const [expandedToolIndex, setExpandedToolIndex] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [conversation]);

  const handleSubmit = (event) => {
    event.preventDefault();
    const content = inputRef.current?.value ?? "";
    if (!content.trim() || loading || resetting) return;
    onSubmit(content.trim());
    inputRef.current.value = "";
    inputRef.current.focus();
  };

  const handleClear = () => {
    onClear();
    inputRef.current?.focus();
  };

  useEffect(() => {
    if (!toolCallHistory.length) {
      setShowToolHistory(false);
      setExpandedToolIndex(null);
    }
  }, [toolCallHistory.length]);

  const toolIndicatorActive = loading || !toolStatusComplete;
  const lastCompletedTool =
    toolCallHistory.length > 0 ? toolCallHistory[toolCallHistory.length - 1] : null;
  const displayedTool = toolIndicatorActive
    ? latestToolCall || lastCompletedTool
    : lastCompletedTool;
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "";
    try {
      return new Date(timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch (err) {
      return timestamp;
    }
  };
  const displayedName = displayedTool?.name;
  const displayedTime = displayedTool?.timestamp
    ? formatTimestamp(displayedTool.timestamp)
    : null;
  const formatDuration = (duration) => {
    if (typeof duration !== "number" || Number.isNaN(duration)) return "—";
    if (duration < 1000) return `${duration.toFixed(0)} ms`;
    return `${(duration / 1000).toFixed(2)} s`;
  };
  const stringifyArgs = (args) => {
    if (!args || typeof args !== "object") return "None";
    try {
      return JSON.stringify(args, null, 2);
    } catch (err) {
      return String(args);
    }
  };
  const toolIndicatorText = displayedName
    ? `Last Tool: ${displayedName}${displayedTime ? ` • ${displayedTime}` : ""}`
    : toolIndicatorActive
    ? "Waiting for tool call…"
    : "No tools called";
  const toolIndicatorClassName = [
    "chat-header__tool-indicator",
    toolIndicatorActive ? "chat-header__tool-indicator--live" : "",
    toolCallHistory.length ? "chat-header__tool-indicator--clickable" : "",
    showToolHistory ? "chat-header__tool-indicator--open" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const handleToolIndicatorClick = () => {
    if (!toolCallHistory.length) return;
    setShowToolHistory((prev) => !prev);
  };

  const handleHistoryItemClick = (index) => {
    setExpandedToolIndex((prev) => (prev === index ? null : index));
  };

  const startRecording = async () => {
    try {
      setRecordingError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm") 
          ? "audio/webm" 
          : MediaRecorder.isTypeSupported("audio/mp4")
          ? "audio/mp4"
          : "audio/ogg"
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: mediaRecorder.mimeType || "audio/webm" 
        });
        
        // Stop all tracks to release microphone
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        // Send to backend for transcription
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");
        
        try {
          const response = await fetch(`${backendUrl}/transcribe`, {
            method: "POST",
            body: formData,
          });
          
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Transcription failed" }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
          }
          
          const data = await response.json();
          if (data.text && data.text.trim()) {
            const transcribedText = data.text.trim();
            // Set the transcribed text in the input field
            if (inputRef.current) {
              inputRef.current.value = transcribedText;
            }
            // Automatically submit the transcribed message
            onSubmit(transcribedText);
            // Clear the input field after submission (same as handleSubmit does)
            if (inputRef.current) {
              inputRef.current.value = "";
            }
          } else {
            setRecordingError("No speech detected. Please try again.");
          }
        } catch (err) {
          console.error("Transcription error:", err);
          setRecordingError(err.message || "Failed to transcribe audio. Please try again.");
        } finally {
          setIsRecording(false);
        }
      };
      
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setRecordingError("Recording error occurred. Please try again.");
        setIsRecording(false);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setRecordingError(
        err.name === "NotAllowedError" 
          ? "Microphone access denied. Please allow microphone access and try again."
          : err.name === "NotFoundError"
          ? "No microphone found. Please connect a microphone and try again."
          : "Failed to access microphone. Please try again."
      );
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
      
      // Stop stream if still active
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return (
    <section className="chat-pane">
      <div className="chat-box">
        <header className="chat-header">
          <div className="chat-header__left">
            <button
              type="button"
              className="chat-header__toggle"
              onClick={onToggleSidebar}
              aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? "◀" : "▶"}
            </button>
            <span className="chat-header__title">{useCaseLabel || "Conversation"}</span>
          </div>
          <div className="chat-header__right">
            <div className="chat-header__tool-wrapper">
              <button
                type="button"
                className={toolIndicatorClassName}
                onClick={handleToolIndicatorClick}
                disabled={!toolCallHistory.length}
                aria-expanded={showToolHistory}
              >
                {toolIndicatorText}
              </button>
              {showToolHistory && toolCallHistory.length > 0 && (
                <div className="chat-header__tool-history" role="list">
                  {toolCallHistory.map((tool, index) => (
                    <div key={`${tool?.name ?? "tool"}-${index}`} className="chat-header__tool-history-entry">
                      <button
                        type="button"
                        className={`chat-header__tool-history-item ${expandedToolIndex === index ? "chat-header__tool-history-item--expanded" : ""}`}
                        onClick={() => handleHistoryItemClick(index)}
                        role="listitem"
                        aria-expanded={expandedToolIndex === index}
                      >
                        <span className="chat-header__tool-history-index">{index + 1}.</span>
                        <div className="chat-header__tool-history-detail">
                          <span className="chat-header__tool-history-name">{tool?.name ?? "Unknown tool"}</span>
                          {tool?.timestamp && (
                            <span className="chat-header__tool-history-time">
                              {formatTimestamp(tool.timestamp)}
                            </span>
                          )}
                        </div>
                      </button>
                      {expandedToolIndex === index && (
                        <div className="chat-header__tool-history-details">
                          <div className="chat-header__tool-history-details-row">
                            <div className="chat-header__tool-history-label-row">
                              <span className="chat-header__tool-history-label">Args</span>
                              <button
                                type="button"
                                className="chat-header__tool-history-copy"
                                onClick={() => {
                                  navigator.clipboard.writeText(stringifyArgs(tool?.args));
                                }}
                              >
                                Copy
                              </button>
                            </div>
                            <pre className="chat-header__tool-history-code">
                              {stringifyArgs(tool?.args)}
                            </pre>
                          </div>
                          <div className="chat-header__tool-history-details-row">
                            <div className="chat-header__tool-history-label-row">
                              <span className="chat-header__tool-history-label">Response</span>
                              <button
                                type="button"
                                className="chat-header__tool-history-copy"
                                onClick={() => {
                                  navigator.clipboard.writeText(tool?.response || "");
                                }}
                              >
                                Copy
                              </button>
                            </div>
                            <pre className="chat-header__tool-history-code">
                              {tool?.response ? tool.response : "No response"}
                            </pre>
                          </div>
                          <div className="chat-header__tool-history-details-row chat-header__tool-history-details-row--inline">
                            <span className="chat-header__tool-history-label">Response Time</span>
                            <span className="chat-header__tool-history-response-time">
                              {formatDuration(tool?.duration_ms)}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              className="chat-header__clear"
              onClick={handleClear}
              disabled={loading || resetting}
            >
              {resetting ? "Clearing…" : "Clear"}
            </button>
          </div>
        </header>

        <div className="chat-messages">
          {conversation.length === 0 && (
            <div className="chat-empty">Start the conversation below</div>
          )}

          {conversation.map((msg, index) => (
            <div
              key={`${index}-${msg.isUser ? "user" : "assistant"}`}
              className={`chat-message ${msg.isUser ? "chat-message--user" : "chat-message--assistant"}`}
            >
              {msg.isUser ? (
                msg.text
              ) : (
                <span
                  dangerouslySetInnerHTML={{
                    __html: msg.rendered ?? "",
                  }}
                />
              )}
            </div>
          ))}

          {loading && <div className="chat-thinking">Thinking…</div>}
          <div ref={messagesEndRef} />
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            className="chat-input"
            placeholder="Type your message…"
            disabled={resetting || isRecording}
            ref={inputRef}
          />
          <button
            type="button"
            className={`chat-button-voice ${isRecording ? "chat-button-voice--recording" : ""}`}
            onClick={toggleRecording}
            disabled={loading || resetting}
            aria-label={isRecording ? "Stop recording" : "Start voice input"}
            title={isRecording ? "Stop recording" : "Start voice input"}
          >
            {isRecording ? (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="chat-button-voice__icon"
              >
                <rect
                  x="7"
                  y="7"
                  width="10"
                  height="10"
                  rx="1.5"
                  fill="currentColor"
                />
              </svg>
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="chat-button-voice__icon"
              >
                <path
                  d="M12 2C10.3431 2 9 3.34315 9 5V12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12V5C15 3.34315 13.6569 2 12 2Z"
                  fill="currentColor"
                />
                <path
                  d="M19 10V12C19 15.866 15.866 19 12 19C8.13401 19 5 15.866 5 12V10H7V12C7 14.7614 9.23858 17 12 17C14.7614 17 17 14.7614 17 12V10H19Z"
                  fill="currentColor"
                />
                <path
                  d="M11 22H13V20H11V22Z"
                  fill="currentColor"
                />
              </svg>
            )}
          </button>
          <button
            type="submit"
            className="chat-button"
            disabled={loading || resetting || isRecording}
          >
            Send
          </button>
        </form>

        {error && <div className="chat-error">{error}</div>}
        {recordingError && <div className="chat-error">{recordingError}</div>}
      </div>
    </section>
  );
}

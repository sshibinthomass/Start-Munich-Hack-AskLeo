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
}) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [showToolHistory, setShowToolHistory] = useState(false);
  const [expandedToolIndex, setExpandedToolIndex] = useState(null);

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
            disabled={resetting}
            ref={inputRef}
          />
          <button
            type="submit"
            className="chat-button"
            disabled={loading || resetting}
          >
            Send
          </button>
        </form>

        {error && <div className="chat-error">{error}</div>}
      </div>
    </section>
  );
}

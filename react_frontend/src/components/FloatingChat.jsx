import React, { useState } from "react";
import { ChatWindow } from "./ChatWindow";
import "./FloatingChat.css";

export function FloatingChat({
  conversation,
  onSubmit,
  onClear,
  loading,
  resetting,
  error,
  useCaseLabel,
  latestToolCall,
  toolCallHistory,
  toolStatusComplete,
  backendUrl,
  voiceOutputEnabled,
  onVoiceOutputToggle,
  isAudioPlaying,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [showToolHistory, setShowToolHistory] = useState(false);
  const [expandedToolIndex, setExpandedToolIndex] = useState(null);

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

  const toolIndicatorActive = loading || !toolStatusComplete;
  const lastCompletedTool =
    toolCallHistory.length > 0 ? toolCallHistory[toolCallHistory.length - 1] : null;
  const displayedTool = toolIndicatorActive
    ? latestToolCall || lastCompletedTool
    : lastCompletedTool;
  const displayedName = displayedTool?.name;
  const displayedTime = displayedTool?.timestamp
    ? formatTimestamp(displayedTool.timestamp)
    : null;
  const toolIndicatorText = displayedName
    ? `${displayedName}${displayedTime ? ` • ${displayedTime}` : ""}`
    : toolIndicatorActive
    ? "Waiting for tool call…"
    : "No tools called";

  const toggleChat = () => {
    if (isOpen) {
      setIsMinimized(!isMinimized);
    } else {
      setIsOpen(true);
      setIsMinimized(false);
    }
  };

  const closeChat = () => {
    setIsOpen(false);
    setIsMinimized(false);
  };

  return (
    <div className="floating-chat">
      {isOpen && (
        <div className={`floating-chat__window ${isMinimized ? "floating-chat__window--minimized" : ""}`}>
          <div className="floating-chat__header">
            <div className="floating-chat__header-left">
              <span className="floating-chat__title">{useCaseLabel || "Chat Assistant"}</span>
              {toolCallHistory.length > 0 && (
                <button
                  type="button"
                  className={`floating-chat__tool-indicator ${toolIndicatorActive ? "floating-chat__tool-indicator--active" : ""}`}
                  onClick={() => setShowToolHistory(!showToolHistory)}
                  title={toolIndicatorText}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                    <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                    <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                  </svg>
                  <span className="floating-chat__tool-count">{toolCallHistory.length}</span>
                </button>
              )}
            </div>
            <div className="floating-chat__controls">
              <button
                type="button"
                className="floating-chat__clear"
                onClick={onClear}
                disabled={loading || resetting || conversation.length === 0}
                title="Clear conversation"
                aria-label="Clear conversation"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                </svg>
              </button>
              <button
                type="button"
                className="floating-chat__minimize"
                onClick={() => setIsMinimized(!isMinimized)}
                aria-label={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? "□" : "−"}
              </button>
              <button
                type="button"
                className="floating-chat__close"
                onClick={closeChat}
                aria-label="Close chat"
              >
                ×
              </button>
            </div>
          </div>
          {!isMinimized && showToolHistory && toolCallHistory.length > 0 && (
            <div className="floating-chat__tool-history">
              <div className="floating-chat__tool-history-header">
                <span>Tool Calls ({toolCallHistory.length})</span>
                <button
                  type="button"
                  className="floating-chat__tool-history-close"
                  onClick={() => setShowToolHistory(false)}
                  aria-label="Close tool history"
                >
                  ×
                </button>
              </div>
              <div className="floating-chat__tool-history-list">
                {toolCallHistory.map((tool, index) => (
                  <div key={`${tool?.name ?? "tool"}-${index}`} className="floating-chat__tool-history-item">
                    <button
                      type="button"
                      className={`floating-chat__tool-history-item-button ${expandedToolIndex === index ? "floating-chat__tool-history-item-button--expanded" : ""}`}
                      onClick={() => setExpandedToolIndex(expandedToolIndex === index ? null : index)}
                    >
                      <span className="floating-chat__tool-history-index">{index + 1}.</span>
                      <span className="floating-chat__tool-history-name">{tool?.name ?? "Unknown tool"}</span>
                      {tool?.timestamp && (
                        <span className="floating-chat__tool-history-time">{formatTimestamp(tool.timestamp)}</span>
                      )}
                    </button>
                    {expandedToolIndex === index && (
                      <div className="floating-chat__tool-history-details">
                        <div className="floating-chat__tool-history-detail-row">
                          <div className="floating-chat__tool-history-label-row">
                            <span className="floating-chat__tool-history-label">Arguments:</span>
                            <button
                              type="button"
                              className="floating-chat__tool-history-copy"
                              onClick={() => navigator.clipboard.writeText(stringifyArgs(tool?.args))}
                            >
                              Copy
                            </button>
                          </div>
                          <pre className="floating-chat__tool-history-code">{stringifyArgs(tool?.args)}</pre>
                        </div>
                        <div className="floating-chat__tool-history-detail-row">
                          <div className="floating-chat__tool-history-label-row">
                            <span className="floating-chat__tool-history-label">Response:</span>
                            <button
                              type="button"
                              className="floating-chat__tool-history-copy"
                              onClick={() => navigator.clipboard.writeText(tool?.response || "")}
                            >
                              Copy
                            </button>
                          </div>
                          <pre className="floating-chat__tool-history-code">{tool?.response || "No response"}</pre>
                        </div>
                        <div className="floating-chat__tool-history-detail-row floating-chat__tool-history-detail-row--inline">
                          <span className="floating-chat__tool-history-label">Response Time:</span>
                          <span>{formatDuration(tool?.duration_ms)}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          {!isMinimized && (
            <div className="floating-chat__content">
              <ChatWindow
                conversation={conversation}
                onSubmit={onSubmit}
                onClear={onClear}
                loading={loading}
                resetting={resetting}
                error={error}
                useCaseLabel={useCaseLabel}
                sidebarOpen={false}
                onToggleSidebar={null}
                latestToolCall={latestToolCall}
                toolCallHistory={toolCallHistory}
                toolStatusComplete={toolStatusComplete}
                backendUrl={backendUrl}
                voiceOutputEnabled={voiceOutputEnabled}
                onVoiceOutputToggle={onVoiceOutputToggle}
                isAudioPlaying={isAudioPlaying}
              />
            </div>
          )}
        </div>
      )}
      {!isOpen && (
        <button
          type="button"
          className="floating-chat__button"
          onClick={toggleChat}
          aria-label="Open chat"
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2ZM20 16H6L4 18V4H20V16Z"
              fill="currentColor"
            />
            <path
              d="M7 9H17V11H7V9ZM7 12H14V14H7V12Z"
              fill="currentColor"
            />
          </svg>
        </button>
      )}
    </div>
  );
}


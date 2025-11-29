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
            <span className="floating-chat__title">{useCaseLabel || "Chat Assistant"}</span>
            <div className="floating-chat__controls">
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


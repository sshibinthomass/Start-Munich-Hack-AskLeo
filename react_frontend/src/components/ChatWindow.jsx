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
  voiceOutputEnabled = false,
  onVoiceOutputToggle = null,
  isAudioPlaying = false,
}) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [showToolHistory, setShowToolHistory] = useState(false);
  const [expandedToolIndex, setExpandedToolIndex] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [continuousMode, setContinuousMode] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const autoRestartTimerRef = useRef(null);
  const silenceCheckIntervalRef = useRef(null);
  const isRecordingRef = useRef(false); // Use ref for reliable state checking in intervals

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

  const startRecording = async (autoStart = false) => {
    try {
      setRecordingError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Set up audio analysis for silence detection (only in continuous mode)
      if (continuousMode) {
        try {
          audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
          analyserRef.current = audioContextRef.current.createAnalyser();
          analyserRef.current.fftSize = 2048;
          analyserRef.current.smoothingTimeConstant = 0.8;
          const source = audioContextRef.current.createMediaStreamSource(stream);
          source.connect(analyserRef.current);
        } catch (audioErr) {
          console.warn("Audio context setup failed:", audioErr);
          // Continue without silence detection if audio context fails
        }
      }
      
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
        
        // Clean up audio context
        if (audioContextRef.current) {
          try {
            audioContextRef.current.close();
          } catch (e) {
            // Ignore errors when closing
          }
          audioContextRef.current = null;
          analyserRef.current = null;
        }
        
        // Clear silence detection timers
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        if (silenceCheckIntervalRef.current) {
          clearInterval(silenceCheckIntervalRef.current);
          silenceCheckIntervalRef.current = null;
        }
        
        // Only process if we have audio chunks and blob has content
        if (audioChunksRef.current.length > 0 && audioBlob.size > 0) {
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
              // Automatically submit the transcribed message
              onSubmit(transcribedText);
              // In continuous mode, auto-restart will happen after response completes
            } else if (!autoStart && !continuousMode) {
              setRecordingError("No speech detected. Please try again.");
            }
          } catch (err) {
            console.error("Transcription error:", err);
            if (!autoStart && !continuousMode) {
              setRecordingError(err.message || "Failed to transcribe audio. Please try again.");
            }
          }
        }
        
        setIsRecording(false);
        isRecordingRef.current = false;
      };
      
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setRecordingError("Recording error occurred. Please try again.");
        setIsRecording(false);
        isRecordingRef.current = false;
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        // Clean up audio context
        if (audioContextRef.current) {
          try {
            audioContextRef.current.close();
          } catch (e) {}
          audioContextRef.current = null;
          analyserRef.current = null;
        }
      };
      
      // Start recording with timeslice for continuous mode (better for silence detection)
      if (continuousMode) {
        mediaRecorder.start(100); // Collect data every 100ms
        setIsRecording(true);
        isRecordingRef.current = true;
        // Start silence detection after a short delay to ensure audio context is ready
        setTimeout(() => {
          if (isRecordingRef.current && continuousMode && analyserRef.current) {
            console.log("Starting silence detection...");
            startSilenceDetection();
          } else {
            console.log("Silence detection not started:", {
              isRecording: isRecordingRef.current,
              continuousMode,
              hasAnalyser: !!analyserRef.current
            });
          }
        }, 500); // Wait 500ms for everything to initialize
      } else {
        mediaRecorder.start();
        setIsRecording(true);
        isRecordingRef.current = true;
      }
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

  // Silence detection function
  const startSilenceDetection = () => {
    if (!analyserRef.current) {
      console.error("Analyser not available for silence detection");
      return;
    }
    
    // Clear any existing interval
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
      silenceCheckIntervalRef.current = null;
    }
    
    let silenceStartTime = null;
    let hasDetectedSpeech = false;
    let speechStartTime = null;
    const SILENCE_THRESHOLD = 0.01; // Lower threshold for better sensitivity
    const SILENCE_DURATION = 1500; // 1.5 seconds of silence to stop
    const MIN_SPEECH_DURATION = 200; // Minimum 200ms of speech
    
    console.log("Silence detection started with threshold:", SILENCE_THRESHOLD);
    
    silenceCheckIntervalRef.current = setInterval(() => {
      // Use ref instead of state for reliable checking
      if (!analyserRef.current || !isRecordingRef.current || !continuousMode) {
        console.log("Stopping silence detection - conditions not met");
        if (silenceCheckIntervalRef.current) {
          clearInterval(silenceCheckIntervalRef.current);
          silenceCheckIntervalRef.current = null;
        }
        return;
      }
      
      try {
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteTimeDomainData(dataArray);
        
        // Calculate RMS volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const normalized = (dataArray[i] - 128) / 128;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / bufferLength);
        
        // Log volume occasionally for debugging (every 50 checks = ~5 seconds)
        if (Math.random() < 0.02) {
          console.log("Audio level:", rms.toFixed(4), "Threshold:", SILENCE_THRESHOLD);
        }
        
        if (rms > SILENCE_THRESHOLD) {
          // Sound detected
          if (!hasDetectedSpeech) {
            speechStartTime = Date.now();
            hasDetectedSpeech = true;
            console.log("Speech detected, starting to monitor for silence...");
          }
          silenceStartTime = null; // Reset silence timer
        } else {
          // Silence detected
          if (hasDetectedSpeech) {
            const speechDuration = speechStartTime ? Date.now() - speechStartTime : 0;
            
            if (speechDuration >= MIN_SPEECH_DURATION) {
              if (silenceStartTime === null) {
                silenceStartTime = Date.now();
                console.log("Silence started, waiting", SILENCE_DURATION, "ms...");
              } else {
                const silenceDuration = Date.now() - silenceStartTime;
                if (silenceDuration >= SILENCE_DURATION) {
                  // Stop recording after silence duration
                  console.log("Silence duration reached, stopping recording...");
                  
                  // Clear interval first
                  if (silenceCheckIntervalRef.current) {
                    clearInterval(silenceCheckIntervalRef.current);
                    silenceCheckIntervalRef.current = null;
                  }
                  
                  // Stop the recorder
                  if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
                    console.log("Calling mediaRecorder.stop()...");
                    mediaRecorderRef.current.stop();
                  } else {
                    console.warn("MediaRecorder not in recording state:", mediaRecorderRef.current?.state);
                  }
                  
                  silenceStartTime = null;
                  hasDetectedSpeech = false;
                  return; // Exit the interval
                }
              }
            }
          }
        }
      } catch (err) {
        console.error("Error in silence detection:", err);
      }
    }, 100); // Check every 100ms
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecordingRef.current) {
      console.log("Manual stop recording called");
      if (mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
      isRecordingRef.current = false;
      
      // Stop stream if still active
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      
      // Clean up audio context
      if (audioContextRef.current) {
        try {
          audioContextRef.current.close();
        } catch (e) {
          // Ignore errors
        }
        audioContextRef.current = null;
        analyserRef.current = null;
      }
      
      // Clear timers
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      if (silenceCheckIntervalRef.current) {
        clearInterval(silenceCheckIntervalRef.current);
        silenceCheckIntervalRef.current = null;
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

  // Auto-restart recording after response completes (in continuous mode)
  // But only if audio is not playing
  useEffect(() => {
    if (continuousMode && !loading && !isRecording && !resetting && !isAudioPlaying) {
      // Wait a brief moment before restarting to allow UI to update
      autoRestartTimerRef.current = setTimeout(() => {
        if (continuousMode && !loading && !isRecording && !resetting && !isAudioPlaying) {
          startRecording(true);
        }
      }, 800); // 800ms delay after response completes
    }
    
    return () => {
      if (autoRestartTimerRef.current) {
        clearTimeout(autoRestartTimerRef.current);
        autoRestartTimerRef.current = null;
      }
    };
  }, [loading, continuousMode, isRecording, resetting, isAudioPlaying]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (audioContextRef.current) {
        try {
          audioContextRef.current.close();
        } catch (e) {}
      }
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
      if (silenceCheckIntervalRef.current) {
        clearInterval(silenceCheckIntervalRef.current);
      }
      if (autoRestartTimerRef.current) {
        clearTimeout(autoRestartTimerRef.current);
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
            placeholder={
              isAudioPlaying
                ? "Agent is speaking..."
                : continuousMode
                ? "Continuous mode: Speak naturally..."
                : "Type your message…"
            }
            disabled={resetting || (isRecording && continuousMode) || isAudioPlaying}
            ref={inputRef}
          />
          {onVoiceOutputToggle && (
            <button
              type="button"
              className={`chat-button-tts ${voiceOutputEnabled ? "chat-button-tts--active" : ""}`}
              onClick={onVoiceOutputToggle}
              disabled={loading || resetting || isAudioPlaying}
              title={voiceOutputEnabled ? "Disable voice output" : "Enable voice output (ElevenLabs)"}
              aria-label={voiceOutputEnabled ? "Disable voice output" : "Enable voice output"}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M3 9V15H7L12 20V4L7 9H3ZM16.5 12C16.5 10.23 15.48 8.71 14.5 7.97V16.02C15.48 15.29 16.5 13.77 16.5 12ZM14.5 3.13V5.29C16.89 6.15 18.5 8.83 18.5 12C18.5 15.17 16.89 17.85 14.5 18.71V20.87C18.01 19.93 20.5 16.35 20.5 12C20.5 7.65 18.01 4.07 14.5 3.13Z"
                  fill="currentColor"
                />
              </svg>
            </button>
          )}
          <button
            type="button"
            className={`chat-button-continuous ${continuousMode ? "chat-button-continuous--active" : ""}`}
            onClick={() => {
              if (!continuousMode) {
                setContinuousMode(true);
                if (!isRecording && !loading && !isAudioPlaying) {
                  startRecording(true);
                }
              } else {
                setContinuousMode(false);
                if (isRecording) {
                  stopRecording();
                }
              }
            }}
            disabled={loading || resetting || isAudioPlaying}
            title={
              isAudioPlaying
                ? "Wait for audio to finish"
                : continuousMode
                ? "Disable continuous mode"
                : "Enable continuous voice conversation"
            }
            aria-label={
              isAudioPlaying
                ? "Wait for audio to finish"
                : continuousMode
                ? "Disable continuous mode"
                : "Enable continuous voice conversation"
            }
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
                fill="currentColor"
              />
              {continuousMode && (
                <path
                  d="M12 6C8.69 6 6 8.69 6 12C6 15.31 8.69 18 12 18C15.31 18 18 15.31 18 12C18 8.69 15.31 6 12 6Z"
                  fill="currentColor"
                />
              )}
            </svg>
          </button>
          <button
            type="button"
            className={`chat-button-voice ${isRecording ? "chat-button-voice--recording" : ""}`}
            onClick={toggleRecording}
            disabled={loading || resetting || continuousMode || isAudioPlaying}
            aria-label={isRecording ? "Stop recording" : "Start voice input"}
            title={
              isAudioPlaying
                ? "Wait for audio to finish"
                : continuousMode
                ? "Recording automatically"
                : isRecording
                ? "Stop recording"
                : "Start voice input"
            }
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
            disabled={loading || resetting || isRecording || isAudioPlaying}
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

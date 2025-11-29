import React from "react";

export function Sidebar({
  model,
  models,
  useCase,
  useCases,
  onModelChange,
  backendUrl,
  backendStatus,
  backendStatusMessage,
}) {
  const activeUseCaseLabel =
    useCases.find((option) => option.value === useCase)?.label || "Chat";

  const statusLabel = (() => {
    switch (backendStatus) {
      case "online":
        return "Backend Online";
      case "offline":
        return "Backend Offline";
      case "checking":
      default:
        return "Checking Backend";
    }
  })();

  return (
    <aside className="sidebar">
      <h1 className="sidebar__title">{activeUseCaseLabel}</h1>
      <div className="sidebar__form">
        <label className="sidebar__label">
          Model
          <select
            value={model}
            onChange={(event) => onModelChange(event.target.value)}
            className="sidebar__select"
          >
            {models.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="sidebar__footer">
        <span className={`sidebar__status sidebar__status--${backendStatus}`}>
          <span className="sidebar__status-indicator" />
          {statusLabel}
        </span>
        <div className="sidebar__backend-url">
          <code>{backendUrl}</code>
        </div>
        {backendStatusMessage && (
          <div className="sidebar__footer-message">{backendStatusMessage}</div>
        )}
      </div>
    </aside>
  );
}

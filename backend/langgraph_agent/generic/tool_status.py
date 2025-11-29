import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypedDict


class ToolCallEntry(TypedDict, total=False):
    id: Optional[str]
    name: str
    timestamp: str
    args: Dict[str, Any]
    response: str
    duration_ms: Optional[float]


class ToolStatus(TypedDict, total=False):
    latest_tool_call: Optional[ToolCallEntry]
    tool_calls: List[ToolCallEntry]
    completed: bool


_tool_status_store: Dict[str, ToolStatus] = {}
_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
_QUEUE_MAXSIZE = 25


def _ensure_entry(session_key: str) -> ToolStatus:
    status = _tool_status_store.get(session_key)
    if status is None:
        status = {
            "latest_tool_call": None,
            "tool_calls": [],
            "completed": False,
        }
        _tool_status_store[session_key] = status
    return status


def reset_tool_status(session_key: str) -> None:
    if not session_key:
        return
    _tool_status_store[session_key] = {
        "latest_tool_call": None,
        "tool_calls": [],
        "completed": False,
    }
    _broadcast_status(session_key)


def record_tool_call(session_key: str, entry: Optional[ToolCallEntry]) -> None:
    if not session_key or not entry:
        return
    status = _ensure_entry(session_key)
    status["latest_tool_call"] = entry
    tool_calls = status.get("tool_calls") or []
    tool_calls.append(entry)
    status["tool_calls"] = tool_calls
    status["completed"] = False
    _broadcast_status(session_key)


def finalize_tool_status(
    session_key: str, tool_calls: Optional[List[ToolCallEntry]] = None
) -> None:
    if not session_key:
        return
    status = _ensure_entry(session_key)
    status["latest_tool_call"] = None
    if tool_calls is not None:
        status["tool_calls"] = list(tool_calls)
    status["completed"] = True
    _broadcast_status(session_key)


def get_tool_status(session_key: str) -> ToolStatus:
    if not session_key:
        return {
            "latest_tool_call": None,
            "tool_calls": [],
            "completed": True,
        }
    status = _tool_status_store.get(session_key)
    if status is None:
        return {
            "latest_tool_call": None,
            "tool_calls": [],
            "completed": True,
        }
    return {
        "latest_tool_call": status.get("latest_tool_call"),
        "tool_calls": list(status.get("tool_calls") or []),
        "completed": bool(status.get("completed")),
    }


def clear_tool_status(session_key: str) -> None:
    if not session_key:
        return
    _tool_status_store.pop(session_key, None)
    _broadcast_status(session_key)


def _broadcast_status(session_key: str) -> None:
    if not session_key:
        return
    status_snapshot = get_tool_status(session_key)
    for queue in list(_subscribers.get(session_key, [])):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(status_snapshot)
        except asyncio.QueueFull:
            pass


def notify_tool_status(session_key: str) -> None:
    _broadcast_status(session_key)


async def tool_status_stream(session_key: str):
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _subscribers[session_key].append(queue)
    await queue.put(get_tool_status(session_key))

    try:
        while True:
            status = await queue.get()
            yield status
    finally:
        subscribers = _subscribers.get(session_key, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers and session_key in _subscribers:
            _subscribers.pop(session_key, None)

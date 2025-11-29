import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from langgraph_agent.generic.tool_status import (
    ToolCallEntry,
    finalize_tool_status,
    notify_tool_status,
    record_tool_call,
)


def _coerce_to_ai_message(response: Any) -> AIMessage:
    """
    Ensure the response is returned as an AIMessage instance.
    """
    if isinstance(response, AIMessage):
        return response

    if hasattr(response, "content"):
        return AIMessage(content=getattr(response, "content"))

    if isinstance(response, dict) and "content" in response:
        return AIMessage(content=response["content"])

    if isinstance(response, str):
        return AIMessage(content=response)

    return AIMessage(content=str(response))


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    """
    Normalize tool arguments coming from different tool call formats.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return {"raw": raw_args}
    return raw_args or {}


def _describe_tool_call(tool_call: Any) -> Tuple[str, Dict[str, Any], str]:
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")
    else:
        tool_name = getattr(tool_call, "name", "unknown")
        tool_id = getattr(tool_call, "id", "")
        tool_args = {}
        if hasattr(tool_call, "args"):
            tool_args = _parse_tool_args(tool_call.args)
        if not tool_args and hasattr(tool_call, "__dict__"):
            tool_args = tool_call.__dict__.get("args", {})

    return tool_name, tool_args, tool_id


def _log_tool_call(tool_name: str, tool_args: Dict[str, Any], tool_id: str) -> None:
    """
    Provide consistent logging for tool call details.
    """
    print(f"< TOOL CALL: {tool_name} >")
    if tool_id:
        print(f"Tool ID: {tool_id}")
    if tool_args:
        print(f"Arguments: {tool_args}")
    else:
        print("Arguments: (none)")
    print()


def _extract_message_content(message: Any) -> str:
    if hasattr(message, "content"):
        content = getattr(message, "content")
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)
    if isinstance(message, dict):
        return str(message.get("content", message))
    return str(message)


def _extract_tool_message_id(message: Any) -> Optional[str]:
    possible_attrs = [
        "tool_call_id",
        "id",
    ]
    for attr in possible_attrs:
        if hasattr(message, attr):
            value = getattr(message, attr)
            if value:
                return value
    if hasattr(message, "additional_kwargs"):
        meta = getattr(message, "additional_kwargs") or {}
        if isinstance(meta, dict):
            tool_call_id = meta.get("tool_call_id")
            if tool_call_id:
                return tool_call_id
    if isinstance(message, dict):
        for key in ["tool_call_id", "id"]:
            value = message.get(key)
            if value:
                return value
    return None


async def run_llm_tool_loop(
    llm,
    messages: List[Any],
    tool_node=None,
    tools: Optional[List[Any]] = None,
    max_tool_iterations: int = 10,
    session_key: Optional[str] = None,
) -> Dict[str, List[AIMessage]]:
    """
    Execute a loop that alternates between LLM responses and tool executions
    until a final response without tool calls is produced.

    Args:
        llm: The language model to invoke.
        messages: Conversation history to condition the LLM.
        tool_node: Optional LangGraph ToolNode (deprecated, use tools instead).
        tools: Optional list of tools to execute directly.
        max_tool_iterations: Safety limit to avoid infinite loops.

    Returns:
        A dict containing the final AIMessage list, consistent with node expectations.
    """
    iteration = 0
    working_messages = list(messages)
    tool_call_history: List[ToolCallEntry] = []
    final_response: Optional[AIMessage] = None
    pending_entries: Dict[str, Tuple[ToolCallEntry, datetime]] = {}

    while iteration < max_tool_iterations:
        iteration += 1
        response = llm.invoke(working_messages)
        response = _coerce_to_ai_message(response)

        print(f"LLM Response: {response}")

        if getattr(response, "tool_calls", None):
            print("\n\n< TOOL CALLS DETECTED >\n")
            for index, tool_call in enumerate(response.tool_calls):
                tool_name, tool_args, tool_id = _describe_tool_call(tool_call)
                _log_tool_call(tool_name, tool_args, tool_id)
                timestamp = datetime.now(timezone.utc).isoformat()
                entry_id = tool_id or f"{tool_name}-{iteration}-{index}"
                entry: ToolCallEntry = {
                    "id": entry_id,
                    "name": tool_name,
                    "timestamp": timestamp,
                    "args": tool_args
                    if isinstance(tool_args, dict)
                    else {"value": tool_args},
                    "response": "",
                    "duration_ms": None,
                }
                tool_call_history.append(entry)
                if session_key:
                    record_tool_call(session_key, entry)
                pending_entries[entry_id] = (entry, datetime.now(timezone.utc))

            working_messages.append(response)

            # Execute tools directly if available
            tool_messages = None
            if tools:
                tool_call_start = datetime.now(timezone.utc)
                tool_messages = []

                # Create a tool map for quick lookup
                tool_map = {tool.name: tool for tool in tools}

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name, tool_args, tool_id = _describe_tool_call(tool_call)

                    if tool_name in tool_map:
                        tool = tool_map[tool_name]
                        try:
                            # Execute the tool (handle both sync and async)
                            if hasattr(tool, "ainvoke"):
                                tool_result = await tool.ainvoke(tool_args)
                            elif hasattr(tool, "invoke"):
                                tool_result = tool.invoke(tool_args)
                            else:
                                tool_result = str(tool_args)

                            # Create ToolMessage
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id or tool_name,
                            )
                            tool_messages.append(tool_message)
                        except Exception as e:
                            # Create error message
                            tool_message = ToolMessage(
                                content=f"Error executing tool {tool_name}: {str(e)}",
                                tool_call_id=tool_id or tool_name,
                            )
                            tool_messages.append(tool_message)
                    else:
                        # Tool not found
                        tool_message = ToolMessage(
                            content=f"Tool {tool_name} not found",
                            tool_call_id=tool_id or tool_name,
                        )
                        tool_messages.append(tool_message)

                print(f"Tool Response: {tool_messages}")
                tool_call_end = datetime.now(timezone.utc)
            elif tool_node:
                # Fallback to ToolNode if provided (for backward compatibility)
                tool_call_start = datetime.now(timezone.utc)
                tool_state = {"messages": working_messages}
                # Try to invoke without config first, then with config if needed
                try:
                    tool_result = await tool_node.ainvoke(tool_state)
                except ValueError:
                    # If config is required, try with a minimal config
                    config = {"configurable": {"thread_id": session_key or "default"}}
                    tool_result = await tool_node.ainvoke(tool_state, config=config)
                tool_messages = tool_result.get("messages", [])
                print(f"Tool Response: {tool_messages}")
                tool_call_end = datetime.now(timezone.utc)
            else:
                # No tools available, skip tool execution
                continue

            # Process tool messages and add them to working_messages
            # This must run after tool execution regardless of which path was taken
            if tool_messages is not None:
                for tool_message in tool_messages:
                    message_id = _extract_tool_message_id(tool_message)
                    entry_tuple = pending_entries.get(message_id) or (
                        pending_entries.get(getattr(tool_message, "name", None))
                    )
                    response_text = _extract_message_content(tool_message)
                    if entry_tuple:
                        entry, start_time = entry_tuple
                        previous = entry.get("response") or ""
                        entry["response"] = (
                            f"{previous}\n{response_text}".strip()
                            if previous
                            else response_text
                        )
                        duration = (
                            tool_call_end - start_time
                            if isinstance(start_time, datetime)
                            else tool_call_end - tool_call_start
                        )
                        entry["duration_ms"] = max(
                            duration.total_seconds() * 1000.0, 0.0
                        )
                        entry_id = entry.get("id")
                        if entry_id and entry_id in pending_entries:
                            pending_entries.pop(entry_id, None)
                    else:
                        # If we cannot match, append a synthetic entry for transparency
                        timestamp = datetime.now(timezone.utc).isoformat()
                        synthetic_entry: ToolCallEntry = {
                            "id": _extract_tool_message_id(tool_message)
                            or f"tool-message-{timestamp}",
                            "name": getattr(tool_message, "name", "unknown"),
                            "timestamp": timestamp,
                            "args": {},
                            "response": response_text,
                            "duration_ms": (
                                tool_call_end - tool_call_start
                            ).total_seconds()
                            * 1000.0,
                        }
                        tool_call_history.append(synthetic_entry)
                        if session_key:
                            record_tool_call(session_key, synthetic_entry)
                        entry = synthetic_entry

                    if session_key and entry:
                        notify_tool_status(session_key)

                working_messages.extend(tool_messages)

            continue

        final_response = response
        break

    if final_response is None:
        final_response = response

    if session_key:
        finalize_tool_status(session_key, tool_call_history)

    result: Dict[str, Any] = {"messages": [final_response]}
    if tool_call_history:
        result["tool_calls"] = tool_call_history
    else:
        result["tool_calls"] = []
    return result

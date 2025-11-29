import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.graphs.graph_builder import GraphBuilder  # noqa: E402
from langgraph_agent.llms.groq_llm import GroqLLM  # noqa: E402
from langgraph_agent.llms.openai_llm import OpenAiLLM  # noqa: E402
from langgraph_agent.llms.gemini_llm import GeminiLLM  # noqa: E402
from langgraph_agent.llms.ollama_llm import OllamaLLM  # noqa: E402
from langgraph_agent.llms.anthropic_llm import AnthropicLLM  # noqa: E402
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)  # noqa: E402
from langgraph_agent.mcps.tool_loader import load_tools_with_timeout  # noqa: E402
from langgraph_agent.generic import (  # noqa: E402
    reset_tool_status,
    finalize_tool_status,
    get_tool_status,
    clear_tool_status,
    tool_status_stream,
)

# Load environment variables
load_dotenv()

# Global chatbot graph instance
chatbot_graph = None
# Global MCP tools (loaded once at startup)
mcp_tools = None
# In-memory session store: (session_id, use_case) -> list of LangChain messages
session_store: Dict[str, List] = {}


async def load_mcp_tools():
    """
    Load MCP tools once at startup. This function caches the tools
    so they're only loaded once and reused for all requests.
    Returns the list of tools that can be reused.
    """
    global mcp_tools
    if mcp_tools is not None:
        return mcp_tools  # Return cached tools if already loaded

    try:
        # Load tools using the function from mcp_chatbot_node
        tools = await load_tools_with_timeout()

        mcp_tools = tools
        print(f"MCP tools loaded: {len(mcp_tools)} tools")

        return mcp_tools
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []


async def initialize_chatbot():
    """Initialize the chatbot graph with Groq LLM"""
    global chatbot_graph
    try:
        user_controls_input = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "selected_llm": "openai/gpt-oss-20b",
        }
        llm = GroqLLM(user_controls_input)
        llm = llm.get_base_llm()
        graph_builder = GraphBuilder(llm, user_controls_input)
        chatbot_graph = await graph_builder.setup_graph("mcp_chatbot")
        return True
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    # Load MCP tools once at startup
    await load_mcp_tools()

    if not await initialize_chatbot():
        print(
            "Warning: Failed to initialize chatbot. API will still work but chatbot endpoints may fail."
        )
    yield
    # Shutdown (if needed, add cleanup code here)
    # For example: cleanup resources, close connections, etc.


# Initialize FastAPI app
app = FastAPI(
    title="Agentic Base React Backend",
    description="FastAPI backend for the Agentic Base React application",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolCall(BaseModel):
    id: Optional[str] = None
    name: str
    timestamp: str
    args: Dict[str, Any] = Field(default_factory=dict)
    response: str = ""
    duration_ms: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    latest_tool_call: Optional[ToolCall] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ToolStatusResponse(BaseModel):
    latest_tool_call: Optional[ToolCall] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    completed: bool = True


class SimpleChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    provider: Optional[str] = "groq"  # groq | openai | gemini | ollama
    selected_llm: Optional[str] = None
    use_case: Optional[str] = "mcp_chatbot"


class ResetChatRequest(BaseModel):
    session_id: Optional[str] = "default"
    use_case: Optional[str] = "mcp_chatbot"


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic Base React Backend API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "chatbot_initialized": chatbot_graph is not None}


@app.get("/tool-status", response_model=ToolStatusResponse)
async def read_tool_status(
    session_id: Optional[str] = "default",
    use_case: Optional[str] = "mcp_chatbot",
):
    session_key = f"{session_id or 'default'}::{use_case or 'mcp_chatbot'}"
    status = get_tool_status(session_key)
    return ToolStatusResponse(
        latest_tool_call=status.get("latest_tool_call"),
        tool_calls=status.get("tool_calls", []),
        completed=status.get("completed", True),
    )


@app.get("/tool-status/stream")
async def stream_tool_status(
    session_id: Optional[str] = "default",
    use_case: Optional[str] = "mcp_chatbot",
):
    session_key = f"{session_id or 'default'}::{use_case or 'mcp_chatbot'}"

    async def event_generator():
        async for status in tool_status_stream(session_key):
            yield f"data: {json.dumps(status)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat")
async def chat_simple(request: SimpleChatRequest):
    """
    Simple chat endpoint that takes a message.
    Conversation history is maintained on the backend per session_id.
    """
    if chatbot_graph is None:
        # Even if global init failed, we can still serve requests if provider creds are valid
        # so don't hard error here.
        pass

    try:
        # Choose LLM based on provider/model from request
        provider = (request.provider or "groq").lower()
        selected_llm = request.selected_llm
        if provider == "groq":
            user_controls_input = {
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
                "selected_llm": selected_llm or "openai/gpt-oss-20b",
            }
            llm = GroqLLM(user_controls_input).get_base_llm()
        elif provider == "openai":
            user_controls_input = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "selected_llm": selected_llm or "gpt-4o-mini",
            }
            llm = OpenAiLLM(user_controls_input).get_base_llm()
        elif provider == "gemini":
            user_controls_input = {
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
                "selected_llm": selected_llm or "gemini-2.5-flash",
            }
            llm = GeminiLLM(user_controls_input).get_base_llm()
        elif provider == "ollama":
            user_controls_input = {
                "selected_llm": selected_llm or "gemma3:1b",
                "OLLAMA_BASE_URL": os.getenv(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
            }
            llm = OllamaLLM(user_controls_input).get_base_llm()
        elif provider == "anthropic":
            user_controls_input = {
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "selected_llm": selected_llm or "claude-haiku-4-5-20251001",
            }
            llm = AnthropicLLM(user_controls_input).get_base_llm()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}"
            )

        use_case = request.use_case or "mcp_chatbot"

        # Build a lightweight graph for this request with the chosen LLM
        try:
            graph_builder = GraphBuilder(llm, {"selected_llm": selected_llm or ""})

            # For MCP chatbot, use pre-loaded tools
            tools = None
            if use_case == "mcp_chatbot":
                # Use globally loaded tools (loaded once at startup)
                tools = mcp_tools if mcp_tools is not None else await load_mcp_tools()

            # Graph is created but we'll stream directly from LLM
            _ = await graph_builder.setup_graph(use_case, tools=tools)
        except ValueError as graph_error:
            raise HTTPException(status_code=400, detail=str(graph_error))

        # Resolve session and initialize store if needed
        session_id = request.session_id or "default"
        session_key = f"{session_id}::{use_case}"
        if session_key not in session_store:
            session_store[session_key] = []
        reset_tool_status(session_key)

        # Build messages from stored history and current input
        messages = [SystemMessage(content="You are a helpful and efficient assistant.")]
        messages.extend(session_store[session_key])
        user_msg = HumanMessage(content=request.message)
        messages.append(user_msg)

        # Stream response - handle tool calls then stream final response
        async def generate_stream():
            full_response = ""
            result_tool_calls = []

            try:
                from langgraph_agent.nodes.mcp_chatbot_node import MCPChatbotNode
                from langchain_core.messages import ToolMessage

                # Create the node to access tools and LLM
                mcp_node = MCPChatbotNode(llm, tools=tools)
                working_messages = list(messages)
                iteration = 0
                max_iterations = 10

                # Handle tool calls in a loop, streaming the final response
                while iteration < max_iterations:
                    iteration += 1

                    # Stream the LLM response and accumulate chunks
                    response_content = ""
                    accumulated_chunks = []
                    full_response_obj = None

                    # Stream and accumulate chunks to reconstruct full message
                    async for chunk in mcp_node.llm.astream(working_messages):
                        accumulated_chunks.append(chunk)
                        
                        # Stream content chunks to user
                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            response_content += content
                            full_response += content
                            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                        
                        # Check for tool calls in chunks and notify user
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tool_chunk in chunk.tool_call_chunks:
                                tool_name = tool_chunk.get("name", "") or (tool_chunk.get("name", "") if isinstance(tool_chunk, dict) else getattr(tool_chunk, "name", ""))
                                if tool_name:
                                    yield f"data: {json.dumps({'type': 'tool_call', 'tool_name': tool_name})}\n\n"

                    # Reconstruct full response from accumulated chunks
                    if not accumulated_chunks:
                        # No chunks received - this shouldn't happen, but handle it
                        continue
                    
                    # Merge chunks to get full message with complete tool calls
                    from langchain_core.messages import AIMessage
                    try:
                        # Merge all chunks into a full message
                        # The + operator on AIMessageChunk merges them properly
                        full_response_obj = accumulated_chunks[0]
                        for chunk in accumulated_chunks[1:]:
                            full_response_obj = full_response_obj + chunk
                    except Exception as e:
                        # If merging fails, use the last chunk
                        print(f"Error merging chunks: {e}")
                        full_response_obj = accumulated_chunks[-1]

                    # Check if the response has tool calls
                    # After merging, tool_calls should be available on the merged message
                    has_tool_calls = False
                    tool_calls_list = []
                    if full_response_obj:
                        # Check tool_calls attribute (merged chunks should have this populated)
                        if (
                            hasattr(full_response_obj, "tool_calls")
                            and full_response_obj.tool_calls
                        ):
                            has_tool_calls = True
                            tool_calls_list = full_response_obj.tool_calls
                        # Check response_metadata finish_reason for tool calls
                        elif hasattr(full_response_obj, "response_metadata"):
                            metadata = full_response_obj.response_metadata
                            finish_reason = metadata.get("finish_reason", "") if metadata else ""
                            if finish_reason == "tool_calls":
                                # Finish reason indicates tool calls, but tool_calls might not be in merged chunk
                                # We need to get the full response to extract complete tool calls
                                # This is a fallback when merging doesn't populate tool_calls
                                try:
                                    complete_response = await mcp_node.llm.ainvoke(working_messages)
                                    if hasattr(complete_response, "tool_calls") and complete_response.tool_calls:
                                        has_tool_calls = True
                                        tool_calls_list = complete_response.tool_calls
                                        # If we didn't stream any content but the response has content, stream it now
                                        if not response_content and hasattr(complete_response, "content") and complete_response.content:
                                            content = complete_response.content
                                            response_content = content
                                            full_response += content
                                            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                                except Exception as e:
                                    print(f"Error getting complete response for tool calls: {e}")
                                    has_tool_calls = False

                    # If no tool calls, we're done - break
                    if not has_tool_calls:
                        break

                    # Handle tool calls
                    if tool_calls_list and mcp_node.tools:
                        tool_map = {tool.name: tool for tool in mcp_node.tools}
                        tool_messages = []

                        for tool_call in tool_calls_list:
                            tool_name = (
                                getattr(tool_call, "name", None)
                                or tool_call.get("name")
                                if isinstance(tool_call, dict)
                                else None
                            )
                            tool_id = (
                                getattr(tool_call, "id", None) or tool_call.get("id")
                                if isinstance(tool_call, dict)
                                else None
                            )
                            tool_args = (
                                getattr(tool_call, "args", None)
                                or tool_call.get("args")
                                if isinstance(tool_call, dict)
                                else {}
                            )

                            if tool_name in tool_map:
                                tool = tool_map[tool_name]
                                try:
                                    if hasattr(tool, "ainvoke"):
                                        tool_result = await tool.ainvoke(tool_args)
                                    elif hasattr(tool, "invoke"):
                                        tool_result = tool.invoke(tool_args)
                                    else:
                                        tool_result = str(tool_args)

                                    tool_message = ToolMessage(
                                        content=str(tool_result),
                                        tool_call_id=tool_id or tool_name,
                                    )
                                    tool_messages.append(tool_message)

                                    # Record tool call
                                    from datetime import datetime, timezone

                                    entry = {
                                        "id": tool_id or f"{tool_name}-{iteration}",
                                        "name": tool_name,
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "args": tool_args
                                        if isinstance(tool_args, dict)
                                        else {"value": tool_args},
                                        "response": str(tool_result),
                                        "duration_ms": None,
                                    }
                                    result_tool_calls.append(entry)
                                    from langgraph_agent.generic.tool_status import (
                                        record_tool_call,
                                    )

                                    if session_key:
                                        record_tool_call(session_key, entry)
                                except Exception as e:
                                    tool_message = ToolMessage(
                                        content=f"Error: {str(e)}",
                                        tool_call_id=tool_id or tool_name,
                                    )
                                    tool_messages.append(tool_message)

                        working_messages.append(
                            AIMessage(
                                content=response_content, tool_calls=tool_calls_list
                            )
                        )
                        working_messages.extend(tool_messages)
                    else:
                        break

                latest_tool_call = result_tool_calls[-1] if result_tool_calls else None

                # Persist history for this session (user + assistant)
                session_store[session_key].append(user_msg)
                session_store[session_key].append(AIMessage(content=full_response))

                finalize_tool_status(session_key, result_tool_calls)

                # Send final message with tool calls
                yield f"data: {json.dumps({'type': 'done', 'tool_calls': result_tool_calls, 'latest_tool_call': latest_tool_call})}\n\n"

            except Exception as e:
                # Send error as SSE
                import traceback

                error_msg = str(e)
                print(f"Streaming error: {error_msg}")
                print(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                finalize_tool_status(session_key, [])

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        session_id = request.session_id or "default"
        use_case = request.use_case or "mcp_chatbot"
        session_key = f"{session_id}::{use_case}"
        finalize_tool_status(session_key, [])
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@app.post("/chat/reset")
async def reset_chat(request: ResetChatRequest):
    session_id = request.session_id or "default"
    use_case = request.use_case or "mcp_chatbot"
    session_key = f"{session_id}::{use_case}"
    session_store.pop(session_key, None)
    clear_tool_status(session_key)
    return {"status": "success"}


def main():
    """Main function to run the FastAPI server"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import tempfile
import openai
from elevenlabs.client import ElevenLabs

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
from langgraph_agent.prompts import get_scout_system_prompt  # noqa: E402
from langgraph_agent.generic import (  # noqa: E402
    reset_tool_status,
    finalize_tool_status,
    get_tool_status,
    clear_tool_status,
    tool_status_stream,
)
from langgraph_agent.agent_communication import ExternalAPIAgent  # noqa: E402

# Load environment variables
load_dotenv()

# Global chatbot graph instance
chatbot_graph = None
# Global MCP tools (loaded once at startup)
mcp_tools = None
# In-memory session store: (session_id, use_case) -> list of LangChain messages
session_store: Dict[str, List] = {}
# Store external agent conversation IDs: session_id -> conversation_id
external_agent_conversations: Dict[str, str] = {}


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


class AgentToAgentRequest(BaseModel):
    provider: Optional[str] = "groq"
    selected_llm: Optional[str] = None
    max_exchanges: Optional[int] = 10
    conversation_mode: Optional[str] = "fixed"  # "fixed" or "until_deal"
    voice_output_enabled: Optional[bool] = False


class DunklerChatRequest(BaseModel):
    message: str
    conversation_id: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic Base React Backend API", "status": "running"}


@app.get("/products")
async def get_products():
    """Get all products from the product.json file."""
    try:
        product_file = project_root / "data" / "product.json"
        with open(product_file, "r") as f:
            products = json.load(f)
        return products
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Product data not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid product data format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading products: {str(e)}")


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
        system_prompt = get_scout_system_prompt()
        messages = [SystemMessage(content=system_prompt)]
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
                        if (
                            hasattr(chunk, "tool_call_chunks")
                            and chunk.tool_call_chunks
                        ):
                            for tool_chunk in chunk.tool_call_chunks:
                                tool_name = tool_chunk.get("name", "") or (
                                    tool_chunk.get("name", "")
                                    if isinstance(tool_chunk, dict)
                                    else getattr(tool_chunk, "name", "")
                                )
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
                            finish_reason = (
                                metadata.get("finish_reason", "") if metadata else ""
                            )
                            if finish_reason == "tool_calls":
                                # Finish reason indicates tool calls, but tool_calls might not be in merged chunk
                                # We need to get the full response to extract complete tool calls
                                # This is a fallback when merging doesn't populate tool_calls
                                try:
                                    complete_response = await mcp_node.llm.ainvoke(
                                        working_messages
                                    )
                                    if (
                                        hasattr(complete_response, "tool_calls")
                                        and complete_response.tool_calls
                                    ):
                                        has_tool_calls = True
                                        tool_calls_list = complete_response.tool_calls
                                        # If we didn't stream any content but the response has content, stream it now
                                        if (
                                            not response_content
                                            and hasattr(complete_response, "content")
                                            and complete_response.content
                                        ):
                                            content = complete_response.content
                                            response_content = content
                                            full_response += content
                                            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                                except Exception as e:
                                    print(
                                        f"Error getting complete response for tool calls: {e}"
                                    )
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


async def get_mcp_chatbot_response(
    message: str,
    session_id: str,
    provider: str,
    selected_llm: str,
    use_case: str = "mcp_chatbot",
) -> str:
    """
    Helper function to get a response from the MCP chatbot.
    Returns the full response text.
    """
    # Choose LLM based on provider/model
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
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
        llm = OllamaLLM(user_controls_input).get_base_llm()
    elif provider == "anthropic":
        user_controls_input = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "selected_llm": selected_llm or "claude-haiku-4-5-20251001",
        }
        llm = AnthropicLLM(user_controls_input).get_base_llm()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Build graph with tools
    tools = None
    if use_case == "mcp_chatbot":
        tools = mcp_tools if mcp_tools is not None else await load_mcp_tools()

    graph_builder = GraphBuilder(llm, {"selected_llm": selected_llm or ""})
    _ = await graph_builder.setup_graph(use_case, tools=tools)

    # Get or create session
    session_key = f"{session_id}::{use_case}"
    if session_key not in session_store:
        session_store[session_key] = []
    reset_tool_status(session_key)

    # Build messages
    system_prompt = get_scout_system_prompt()
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(session_store[session_key])
    user_msg = HumanMessage(content=message)
    messages.append(user_msg)

    # Get response using MCPChatbotNode
    from langgraph_agent.nodes.mcp_chatbot_node import MCPChatbotNode
    from langchain_core.messages import ToolMessage

    mcp_node = MCPChatbotNode(llm, tools=tools)
    working_messages = list(messages)
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1

        # Get LLM response
        response_obj = await mcp_node.llm.ainvoke(working_messages)

        # Check for tool calls
        has_tool_calls = False
        tool_calls_list = []
        if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
            has_tool_calls = True
            tool_calls_list = response_obj.tool_calls

        # If no tool calls, we're done
        if not has_tool_calls:
            response_content = (
                response_obj.content
                if hasattr(response_obj, "content")
                else str(response_obj)
            )
            # Update session store
            session_store[session_key].append(user_msg)
            session_store[session_key].append(AIMessage(content=response_content))
            return response_content

        # Handle tool calls
        if tool_calls_list and mcp_node.tools:
            tool_map = {tool.name: tool for tool in mcp_node.tools}
            tool_messages = []

            for tool_call in tool_calls_list:
                tool_name = (
                    getattr(tool_call, "name", None) or tool_call.get("name")
                    if isinstance(tool_call, dict)
                    else None
                )
                tool_id = (
                    getattr(tool_call, "id", None) or tool_call.get("id")
                    if isinstance(tool_call, dict)
                    else None
                )
                tool_args = (
                    getattr(tool_call, "args", None) or tool_call.get("args")
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
                    except Exception as e:
                        tool_message = ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_id or tool_name,
                        )
                        tool_messages.append(tool_message)

            response_content = (
                response_obj.content if hasattr(response_obj, "content") else ""
            )
            working_messages.append(
                AIMessage(content=response_content, tool_calls=tool_calls_list)
            )
            working_messages.extend(tool_messages)
        else:
            break

    # Fallback: return last response
    response_content = (
        response_obj.content if hasattr(response_obj, "content") else str(response_obj)
    )
    session_store[session_key].append(user_msg)
    session_store[session_key].append(AIMessage(content=response_content))
    return response_content


@app.post("/chat/agent-to-agent")
async def agent_to_agent_chat(request: AgentToAgentRequest):
    """
    Agent-to-agent communication endpoint.
    Orchestrates a conversation between the MCP chatbot and external API agent.
    MCP chatbot initiates with "hi" and they exchange messages.
    Can run for a fixed number of exchanges or until a deal is reached.
    Dunkler (external API) always has the last message.
    """
    try:
        provider = (request.provider or "groq").lower()
        selected_llm = request.selected_llm
        conversation_mode = request.conversation_mode or "fixed"
        max_exchanges = (
            request.max_exchanges or 11 if conversation_mode == "fixed" else None
        )

        # Use separate session ID for agent-to-agent
        agent_session_id = f"agent-to-agent-{os.urandom(8).hex()}"

        async def generate_stream():
            try:
                # Initialize external API agent
                external_agent = ExternalAPIAgent()
                await external_agent.initialize_conversation()
                yield f"data: {json.dumps({'type': 'status', 'message': 'Dunkler agent initialized'})}\n\n"

                # Start with MCP chatbot sending "hi"
                initial_message = "hi"
                yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'mcp_chatbot', 'message': initial_message})}\n\n"

                # Helper function to check if a deal has been reached
                def check_deal_reached(messages: List[str]) -> bool:
                    """Check if messages indicate a deal has been reached."""
                    deal_keywords = [
                        "deal",
                        "agreed",
                        "accept",
                        "accepted",
                        "confirmed",
                        "confirmation",
                        "agreement",
                        "settled",
                        "finalized",
                        "approved",
                        "approved",
                        "yes, let's proceed",
                        "sounds good",
                        "we have a deal",
                        "deal is done",
                        "i agree",
                        "i accept",
                        "confirmed",
                        "let's finalize",
                        "agreed upon",
                    ]
                    recent_messages = " ".join(
                        messages[-4:]
                    ).lower()  # Check last 4 messages
                    keyword_count = sum(
                        1 for keyword in deal_keywords if keyword in recent_messages
                    )
                    # If multiple deal keywords appear, likely a deal
                    return keyword_count >= 2

                # Main conversation loop
                # We want Dunkler (external_api) to always have the last message
                # So we do pairs of exchanges: Dunkler -> Leo, and ensure we end with Dunkler
                current_message = initial_message
                exchange_count = 0
                conversation_messages = [initial_message]
                deal_reached = False
                max_iterations = 50  # Safety limit for until_deal mode

                while True:
                    # Check if we should continue
                    if conversation_mode == "fixed":
                        if exchange_count >= max_exchanges:
                            break
                    elif conversation_mode == "until_deal":
                        if deal_reached:
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Deal reached! Conversation ending.'})}\n\n"
                            break
                        if exchange_count >= max_iterations:
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Maximum iterations reached. Ending conversation.'})}\n\n"
                            break
                    # External API agent (Dunkler) responds to current message
                    try:
                        external_response = await external_agent.send_message(
                            current_message
                        )
                        yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'external_api', 'message': external_response})}\n\n"
                        conversation_messages.append(external_response)

                        # Check for deal in until_deal mode
                        if conversation_mode == "until_deal":
                            deal_reached = check_deal_reached(conversation_messages)
                            if deal_reached:
                                break
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'agent': 'external_api', 'error': str(e)})}\n\n"
                        break

                    exchange_count += 1

                    # Check if we've reached the max exchanges (fixed mode) - if so, Dunkler has the last message
                    if conversation_mode == "fixed" and exchange_count >= max_exchanges:
                        break

                    # MCP chatbot (Leo) responds to external API's message
                    try:
                        mcp_response = await get_mcp_chatbot_response(
                            external_response,
                            agent_session_id,
                            provider,
                            selected_llm,
                            "mcp_chatbot",
                        )
                        yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'mcp_chatbot', 'message': mcp_response})}\n\n"
                        current_message = mcp_response
                        conversation_messages.append(mcp_response)

                        # Check for deal in until_deal mode
                        if conversation_mode == "until_deal":
                            deal_reached = check_deal_reached(conversation_messages)
                            if deal_reached:
                                # Let Dunkler have the last word after deal is detected
                                break
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'agent': 'mcp_chatbot', 'error': str(e)})}\n\n"
                        break

                    exchange_count += 1

                # Store the external agent conversation ID for continued chat
                external_agent_conversations[agent_session_id] = (
                    external_agent.conversation_id
                )

                # Clean up MCP chatbot session (but keep external agent conversation)
                session_key = f"{agent_session_id}::mcp_chatbot"
                session_store.pop(session_key, None)
                clear_tool_status(session_key)

                # Final message based on mode
                if conversation_mode == "until_deal" and deal_reached:
                    yield f"data: {json.dumps({'type': 'done', 'exchanges': exchange_count, 'conversation_id': agent_session_id, 'deal_reached': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'done', 'exchanges': exchange_count, 'conversation_id': agent_session_id, 'deal_reached': False})}\n\n"

            except Exception as e:
                import traceback

                error_msg = str(e)
                print(f"Agent-to-agent error: {error_msg}")
                print(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting agent-to-agent chat: {str(e)}"
        )


@app.post("/chat/dunkler")
async def chat_with_dunkler(request: DunklerChatRequest):
    """
    Send a message to Dunkler (external API agent) after agent-to-agent conversation.
    Uses the stored conversation ID to continue the conversation with history.
    """
    try:
        # Get the external agent conversation ID
        if request.conversation_id not in external_agent_conversations:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found. Please start an agent-to-agent conversation first.",
            )

        conversation_id = external_agent_conversations[request.conversation_id]

        # Create external agent instance and set the conversation ID
        external_agent = ExternalAPIAgent()
        external_agent.conversation_id = conversation_id

        # Send message and get response
        response_text = await external_agent.send_message(request.message)

        return {
            "response": response_text,
            "agent": "external_api",
            "conversation_id": request.conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error sending message to Dunkler: {str(e)}"
        )


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel voice
    model_id: Optional[str] = "eleven_turbo_v2_5"  # Fast model for streaming
    stability: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.75


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using ElevenLabs API.
    Returns streaming audio data.
    """
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not configured. Please set it in your .env file.",
        )

    try:
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=elevenlabs_api_key)

        # Use the text_to_speech.convert method which returns a generator
        # Note: The convert method streams audio chunks
        audio_generator = client.text_to_speech.convert(
            voice_id=request.voice_id,
            text=request.text,
            model_id=request.model_id,
            voice_settings={
                "stability": request.stability,
                "similarity_boost": request.similarity_boost,
            },
        )

        # Check if it's actually a generator/iterable
        if not hasattr(audio_generator, "__iter__"):
            raise ValueError("Audio generator is not iterable")

        async def stream_audio():
            try:
                # Iterate through the generator
                for chunk in audio_generator:
                    # The chunks from ElevenLabs are typically bytes
                    if chunk:
                        # If it's already bytes, yield directly
                        if isinstance(chunk, bytes):
                            yield chunk
                        # If it's a response object with data, extract it
                        elif hasattr(chunk, "chunk") and chunk.chunk:
                            yield chunk.chunk
                        elif hasattr(chunk, "data") and chunk.data:
                            yield chunk.data
                        # Try to get bytes from the chunk
                        else:
                            try:
                                chunk_bytes = (
                                    bytes(chunk)
                                    if not isinstance(chunk, bytes)
                                    else chunk
                                )
                                if chunk_bytes:
                                    yield chunk_bytes
                            except (TypeError, ValueError):
                                # If we can't convert, try string representation
                                if isinstance(chunk, str):
                                    yield chunk.encode("utf-8")
            except Exception as stream_error:
                import traceback

                error_trace = traceback.format_exc()
                print(f"Error in audio stream: {stream_error}")
                print(error_trace)
                # Re-raise to be caught by outer exception handler
                raise

        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"ElevenLabs TTS Error: {error_details}")
        # Return more detailed error for debugging
        error_message = f"Error generating speech: {str(e)}"
        if hasattr(e, "__class__"):
            error_message += f" (Type: {e.__class__.__name__})"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using OpenAI Whisper API.
    Accepts audio files in various formats (webm, mp3, wav, etc.)
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Please set it in your .env file.",
        )

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)

    try:
        # Read audio file
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file received")

        # Determine file extension from content type or filename
        file_extension = ".webm"  # default
        if audio.content_type:
            if "webm" in audio.content_type:
                file_extension = ".webm"
            elif "mp3" in audio.content_type:
                file_extension = ".mp3"
            elif "wav" in audio.content_type:
                file_extension = ".wav"
            elif "m4a" in audio.content_type:
                file_extension = ".m4a"
        elif audio.filename:
            # Extract extension from filename
            if "." in audio.filename:
                file_extension = "." + audio.filename.rsplit(".", 1)[1]

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Transcribe with Whisper
            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # Optional: specify language for better accuracy
                )

            transcript_text = (
                transcript.text if hasattr(transcript, "text") else str(transcript)
            )

            return {"text": transcript_text}
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors

    except openai.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key. Please check your OPENAI_API_KEY in .env",
        )
    except openai.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="OpenAI API rate limit exceeded. Please try again later.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error transcribing audio: {str(e)}"
        )


def main():
    """Main function to run the FastAPI server"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

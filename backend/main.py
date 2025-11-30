import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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
from langgraph_agent.tools.custom_tools import send_email, create_event  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
import re  # noqa: E402

try:
    from io import BytesIO  # noqa: E402
    from reportlab.lib.pagesizes import letter, A4  # noqa: E402
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
    from reportlab.lib.units import inch  # noqa: E402
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )  # noqa: E402
    from reportlab.lib import colors  # noqa: E402
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # noqa: E402

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    BytesIO = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    PageBreak = None
    colors = None
    TA_CENTER = None

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
# Store deal information for PDF generation: conversation_id -> deal_data
deal_storage: Dict[str, Dict] = {}


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
    initial_message: Optional[str] = "hello"
    voice_output_enabled: Optional[bool] = False
    personality_traits: Optional[List[str]] = None
    negotiation_strategy: Optional[List[str]] = None
    product_name: Optional[str] = None
    number_of_units: Optional[int] = 5
    min_discount: Optional[float] = 5.0
    max_discount: Optional[float] = 20.0


class BrewBotChatRequest(BaseModel):
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


@app.get("/api/images/{image_name}")
async def get_image(image_name: str):
    """Serve product images from the data directory."""
    try:
        image_path = project_root / "data" / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


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
    personality_traits: Optional[List[str]] = None,
    negotiation_strategy: Optional[List[str]] = None,
    product_name: Optional[str] = None,
    number_of_units: int = 5,
    min_discount: float = 5.0,
    max_discount: float = 20.0,
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
    system_prompt = get_scout_system_prompt(
        personality_traits=personality_traits,
        negotiation_strategy=negotiation_strategy,
        product_name=product_name,
        number_of_units=number_of_units,
        min_discount=min_discount,
        max_discount=max_discount,
    )
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
    BrewBot (external API) always has the last message.
    """
    try:
        provider = (request.provider or "groq").lower()
        selected_llm = request.selected_llm
        conversation_mode = request.conversation_mode or "fixed"
        max_exchanges = (
            request.max_exchanges or 11 if conversation_mode == "fixed" else None
        )
        personality_traits = request.personality_traits
        negotiation_strategy = request.negotiation_strategy
        product_name = request.product_name
        number_of_units = request.number_of_units or 5
        min_discount = request.min_discount or 5.0
        max_discount = request.max_discount or 20.0

        # Use separate session ID for agent-to-agent
        agent_session_id = f"agent-to-agent-{os.urandom(8).hex()}"

        async def generate_stream():
            try:
                # Initialize external API agent
                external_agent = ExternalAPIAgent()
                await external_agent.initialize_conversation()
                yield f"data: {json.dumps({'type': 'status', 'message': 'BrewBot agent initialized'})}\n\n"

                # Start with MCP chatbot sending the initial message
                # If initial message is generic, generate one with configured values
                user_initial_message = request.initial_message or "hello"
                if user_initial_message.lower().strip() in [
                    "hello",
                    "hi",
                    "hey",
                    "start",
                ]:
                    # Auto-generate initial message with configured values
                    initial_message = f"I'm interested in purchasing {number_of_units} units of {product_name}. Could you offer a {max_discount}% discount for this order?"
                else:
                    initial_message = user_initial_message
                yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'mcp_chatbot', 'message': initial_message})}\n\n"

                # Helper function to check if a deal has been reached using Lio's judgment
                async def check_deal_reached(
                    messages: List[str], exchange_count: int = 0
                ) -> bool:
                    """Ask Lio to evaluate if a deal has been reached based on conversation."""
                    # Force deal at 15 exchanges (max limit)
                    if exchange_count >= 15:
                        return True

                    # Don't check too early (need at least 4 exchanges)
                    if exchange_count < 4:
                        return False

                    # Only check every 2 exchanges to reduce API calls (but always check at 14+)
                    if exchange_count < 14 and exchange_count % 2 != 0:
                        return False

                    try:
                        # Create a summary of recent conversation (last 10 messages for better context)
                        recent_messages = (
                            messages[-10:] if len(messages) > 10 else messages
                        )
                        conversation_context = "\n".join(
                            [
                                f"Message {i + 1}: {msg[:200]}..."
                                if len(msg) > 200
                                else f"Message {i + 1}: {msg}"
                                for i, msg in enumerate(recent_messages)
                            ]
                        )

                        # Ask Lio to evaluate if a deal has been reached
                        evaluation_prompt = f"""You are evaluating a negotiation conversation. Based on the following exchange, determine if both parties have reached a mutual agreement or deal.

Important: A deal means BOTH parties have explicitly agreed to specific terms, prices, quantities, or conditions. 
CRITICAL: If a discount of {min_discount}% or better has been agreed upon, the deal is COMPLETE and you must respond "YES".
If the discount is less than {min_discount}%, respond "NO" - the negotiation must continue.

Conversation:
{conversation_context}

Analyze this conversation carefully. Check if:
1. Both parties have explicitly agreed to terms
2. The discount percentage agreed is {min_discount}% or better (best price achieved)

Respond with ONLY one word: "YES" if a clear deal/agreement has been reached with {min_discount}% or better discount, or "NO" if not yet."""

                        # Get Lio's evaluation using a separate session to avoid interfering with main conversation
                        lio_evaluation = await get_mcp_chatbot_response(
                            evaluation_prompt,
                            f"{agent_session_id}_deal_check",
                            provider,
                            selected_llm,
                            "mcp_chatbot",
                            personality_traits=personality_traits,
                            negotiation_strategy=negotiation_strategy,
                            product_name=product_name,
                            number_of_units=number_of_units,
                            min_discount=min_discount,
                            max_discount=max_discount,
                        )

                        # Check if Lio's response indicates a deal
                        lio_response_lower = lio_evaluation.lower().strip()

                        # Look for clear positive indicators
                        if any(
                            word in lio_response_lower
                            for word in [
                                "yes",
                                "deal",
                                "agreed",
                                "agreement",
                                "reached",
                                "finalized",
                            ]
                        ):
                            # Make sure it's not a negative statement
                            if not any(
                                word in lio_response_lower
                                for word in [
                                    "no deal",
                                    "not agreed",
                                    "not reached",
                                    "no agreement",
                                ]
                            ):
                                return True

                        # If we're at 14+ exchanges, be more lenient - if Lio doesn't explicitly say no, consider it a deal
                        if exchange_count >= 14:
                            if "no" not in lio_response_lower or (
                                "no" in lio_response_lower
                                and "not" not in lio_response_lower
                            ):
                                # If response is ambiguous but we're at the limit, lean towards deal
                                if (
                                    len(lio_response_lower) < 50
                                ):  # Short response might be "yes" or similar
                                    return True

                        return False
                    except Exception as e:
                        # If evaluation fails, fall back to forcing deal only at high exchange counts
                        print(f"Error in deal evaluation: {e}")
                        if exchange_count >= 14:
                            return True  # Force deal at 14+ if evaluation fails
                        return False

                # Helper function to check if negotiation has reached a dead-end
                async def check_dead_end(
                    messages: List[str], exchange_count: int = 0
                ) -> tuple[bool, str]:
                    """
                    Ask Lio to evaluate if the negotiation has reached a dead-end.
                    Returns (is_dead_end: bool, reason: str)
                    """
                    # Don't check too early (need at least 6 exchanges to detect patterns)
                    if exchange_count < 6:
                        return False, ""

                    # Check every 3 exchanges after exchange 6 to detect deadlocks
                    if exchange_count < 12 and exchange_count % 3 != 0:
                        return False, ""

                    try:
                        # Create a summary of recent conversation (last 8-10 messages for better context)
                        recent_messages = (
                            messages[-10:] if len(messages) > 10 else messages
                        )
                        conversation_context = "\n".join(
                            [
                                f"Message {i + 1}: {msg[:200]}..."
                                if len(msg) > 200
                                else f"Message {i + 1}: {msg}"
                                for i, msg in enumerate(recent_messages)
                            ]
                        )

                        # Ask Lio to evaluate if there's a dead-end
                        dead_end_prompt = f"""You are evaluating a negotiation conversation to detect if it has reached a dead-end (deadlock) where both parties cannot agree.

A dead-end occurs when:
1. Both parties keep repeating the same offers without progress
2. Multiple rejections without meaningful counter-proposals
3. No movement on discount percentages (stuck at same values)
4. Circular arguments with no resolution
5. Both parties are unwilling to compromise further
6. The negotiation has stalled with no path forward

Conversation:
{conversation_context}

Analyze this conversation carefully. Determine if:
- Both parties are stuck and cannot reach an agreement
- The negotiation has reached an impasse
- There's no reasonable path forward to a deal

Respond in this EXACT format:
- If dead-end detected: "YES: [brief reason why]"
- If negotiation can continue: "NO"

Examples:
- "YES: Both parties stuck at 15% vs 20% discount, no progress for 4 exchanges"
- "YES: Repeated rejections without counter-proposals, negotiation stalled"
- "NO: Still negotiating, progress being made"
- "NO: Parties are making counter-offers and moving towards agreement"

Your response:"""

                        # Get Lio's evaluation using a separate session
                        lio_evaluation = await get_mcp_chatbot_response(
                            dead_end_prompt,
                            f"{agent_session_id}_dead_end_check",
                            provider,
                            selected_llm,
                            "mcp_chatbot",
                            personality_traits=personality_traits,
                            negotiation_strategy=negotiation_strategy,
                            product_name=product_name,
                            number_of_units=number_of_units,
                            min_discount=min_discount,
                            max_discount=max_discount,
                        )

                        # Parse Lio's response
                        lio_response_lower = lio_evaluation.lower().strip()

                        # Check if response indicates a dead-end
                        if lio_response_lower.startswith("yes:"):
                            # Extract the reason
                            reason = (
                                lio_evaluation.split(":", 1)[1].strip()
                                if ":" in lio_evaluation
                                else "Negotiation has reached an impasse with no path forward."
                            )
                            return True, reason
                        elif "yes" in lio_response_lower and any(
                            keyword in lio_response_lower
                            for keyword in [
                                "dead",
                                "stuck",
                                "impasse",
                                "cannot agree",
                                "no progress",
                                "stalled",
                            ]
                        ):
                            # If it says yes with dead-end keywords, extract reason if available
                            if ":" in lio_evaluation:
                                reason = lio_evaluation.split(":", 1)[1].strip()
                            else:
                                reason = "Negotiation has reached an impasse with no path forward."
                            return True, reason

                        return False, ""

                    except Exception as e:
                        # If evaluation fails, don't trigger dead-end (let conversation continue)
                        print(f"Error in dead-end evaluation: {e}")
                        return False, ""

                # Main conversation loop
                # We want BrewBot (external_api) to always have the last message
                # So we do pairs of exchanges: BrewBot -> Lio, and ensure we end with BrewBot
                current_message = initial_message
                exchange_count = 0
                conversation_messages = [initial_message]
                deal_reached = False
                max_iterations = (
                    15  # Aggressive limit: maximum 15 conversations for until_deal mode
                )

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
                            # Force a deal at max iterations
                            deal_reached = True
                            yield f"data: {json.dumps({'type': 'status', 'message': f'Maximum conversations ({max_iterations}) reached. Deal finalized.'})}\n\n"
                            break
                    # External API agent (BrewBot) responds to current message
                    try:
                        external_response = await external_agent.send_message(
                            current_message
                        )
                        yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'external_api', 'message': external_response})}\n\n"
                        conversation_messages.append(external_response)

                        # Check for deal in until_deal mode
                        if conversation_mode == "until_deal":
                            deal_reached = await check_deal_reached(
                                conversation_messages, exchange_count
                            )
                            if deal_reached:
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Lio has determined that a deal has been reached!'})}\n\n"
                                break

                            # Check for dead-end (deadlock) in until_deal mode
                            is_dead_end, dead_end_reason = await check_dead_end(
                                conversation_messages, exchange_count
                            )
                            if is_dead_end:
                                # Store conversation ID for continued chat with BrewBot
                                external_agent_conversations[agent_session_id] = (
                                    external_agent.conversation_id
                                )

                                reason_message = (
                                    f"Negotiation dead-end detected: {dead_end_reason}"
                                    if dead_end_reason
                                    else "Negotiation has reached an impasse with no path forward."
                                )
                                yield f"data: {json.dumps({'type': 'dead_end', 'message': reason_message, 'reason': dead_end_reason or 'No agreement possible'})}\n\n"
                                yield f"data: {json.dumps({'type': 'done', 'exchanges': exchange_count, 'conversation_id': agent_session_id, 'deal_reached': False, 'dead_end': True})}\n\n"
                                break
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'agent': 'external_api', 'error': str(e)})}\n\n"
                        break

                    exchange_count += 1

                    # Check if we've reached the max exchanges (fixed mode) - if so, BrewBot has the last message
                    if conversation_mode == "fixed" and exchange_count >= max_exchanges:
                        break

                    # MCP chatbot (Lio) responds to external API's message
                    try:
                        # Add urgency context if we're approaching the limit
                        message_with_context = external_response
                        if conversation_mode == "until_deal":
                            remaining = max_iterations - exchange_count
                            if remaining <= 3:
                                message_with_context = f"[URGENT: Only {remaining} exchanges remaining. We must finalize the deal now.] {external_response}"
                            elif remaining <= 6:
                                message_with_context = f"[Time is running out: {remaining} exchanges left. Let's work towards closing the deal.] {external_response}"

                        # Add quantity reminder to prevent drift - always remind Lio of the fixed quantity
                        quantity_reminder = f"\n\n⚠️ CRITICAL REMINDER: You are negotiating for EXACTLY {number_of_units} units. NEVER use any other quantity like 5, 10, etc. ALWAYS and ONLY use {number_of_units} units in your response."
                        message_with_context = message_with_context + quantity_reminder

                        mcp_response = await get_mcp_chatbot_response(
                            message_with_context,
                            agent_session_id,
                            provider,
                            selected_llm,
                            "mcp_chatbot",
                            personality_traits=personality_traits,
                            negotiation_strategy=negotiation_strategy,
                            product_name=product_name,
                            number_of_units=number_of_units,
                            min_discount=min_discount,
                            max_discount=max_discount,
                        )
                        yield f"data: {json.dumps({'type': 'agent_message', 'agent': 'mcp_chatbot', 'message': mcp_response})}\n\n"
                        current_message = mcp_response
                        conversation_messages.append(mcp_response)

                        # Check for deal in until_deal mode
                        if conversation_mode == "until_deal":
                            deal_reached = await check_deal_reached(
                                conversation_messages, exchange_count
                            )
                            if deal_reached:
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Lio has determined that a deal has been reached!'})}\n\n"
                                # Let BrewBot have the last word after deal is detected
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

                # If deal is reached, send email and schedule meeting
                if conversation_mode == "until_deal" and deal_reached:
                    try:
                        # Extract vendor email from conversation (look for email patterns)
                        vendor_email = None
                        email_pattern = (
                            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                        )
                        for msg in conversation_messages:
                            emails = re.findall(email_pattern, msg)
                            if emails:
                                vendor_email = emails[0]
                                break

                        # Use default vendor email if not found in conversation
                        if not vendor_email:
                            vendor_email = os.getenv(
                                "VENDOR_EMAIL", "vendor@victoriaarduino.com"
                            )

                        # Create conversation summary (last 6 messages)
                        summary_messages = (
                            conversation_messages[-6:]
                            if len(conversation_messages) > 6
                            else conversation_messages
                        )
                        conversation_summary = "\n".join(
                            [
                                f"- {msg[:200]}..." if len(msg) > 200 else f"- {msg}"
                                for msg in summary_messages
                            ]
                        )

                        # Send confirmation email
                        email_subject = (
                            "Deal Confirmation - Victoria Arduino Espresso Machines"
                        )
                        email_body = f"""Dear Partner,

We are pleased to confirm that we have reached an agreement regarding the Victoria Arduino espresso machines.

Conversation Summary:
{conversation_summary}

We look forward to finalizing the details in our upcoming meeting.

Best regards,
Lio (AI Negotiation Assistant)
"""
                        email_result = send_email(
                            to=vendor_email, subject=email_subject, message=email_body
                        )
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Email sent: {email_result}'})}\n\n"

                        # Schedule meeting for 2 days from now at 10:00 AM UTC
                        meeting_date = datetime.now(timezone.utc) + timedelta(days=2)
                        start_time = meeting_date.replace(
                            hour=10, minute=0, second=0, microsecond=0
                        )
                        end_time = start_time + timedelta(hours=1)

                        meeting_summary = "Deal Finalization Meeting - Victoria Arduino"
                        meeting_description = f"Follow-up meeting to finalize the details of our agreement regarding Victoria Arduino espresso machines.\n\nConversation ID: {agent_session_id}"

                        meeting_result = create_event(
                            summary=meeting_summary,
                            start_time=start_time.isoformat(),
                            end_time=end_time.isoformat(),
                            description=meeting_description,
                            location="Virtual Meeting",
                            attendees=vendor_email,
                        )
                        yield f"data: {json.dumps({'type': 'status', 'message': 'A calendar event has been set to delivery'})}\n\n"

                    except Exception as e:
                        # Don't fail the whole request if email/meeting fails
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Note: Could not send email/schedule meeting: {str(e)}'})}\n\n"

                    # Store deal information for PDF generation
                    try:
                        # Extract negotiated price and discount from conversation
                        negotiated_price = None
                        negotiated_discount = None
                        original_price = None

                        # Ensure conversation_messages is a list of strings
                        conversation_strings = []
                        for msg in conversation_messages:
                            if isinstance(msg, str):
                                conversation_strings.append(msg)
                            else:
                                conversation_strings.append(str(msg))

                        # Try to extract price/discount from conversation
                        for msg in conversation_strings:
                            # Look for discount percentages
                            discount_match = re.search(r"(\d+(?:\.\d+)?)\s*%", msg)
                            if discount_match:
                                try:
                                    disc_value = float(discount_match.group(1))
                                    if (
                                        not negotiated_discount
                                        or disc_value < negotiated_discount
                                    ):
                                        negotiated_discount = disc_value
                                except (ValueError, AttributeError):
                                    pass

                            # Look for price mentions
                            price_match = re.search(
                                r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", msg
                            )
                            if price_match:
                                try:
                                    price_str = price_match.group(1).replace(",", "")
                                    price_value = float(price_str)
                                    if (
                                        not negotiated_price
                                        or price_value < negotiated_price
                                    ):
                                        negotiated_price = price_value
                                except (ValueError, AttributeError):
                                    pass

                        # Get original price from product data
                        try:
                            with open(project_root / "data" / "product.json", "r") as f:
                                products = json.load(f)
                                for product in products:
                                    if product.get("name") == product_name:
                                        for version in product.get("versions", []):
                                            original_price = version.get("price")
                                            break
                                        break
                        except Exception:
                            pass

                        # Store deal data
                        deal_storage[agent_session_id] = {
                            "conversation": conversation_strings,
                            "conversation_id": agent_session_id,
                            "product_name": product_name or "Unknown Product",
                            "number_of_units": number_of_units or 1,
                            "min_discount": min_discount or 5.0,
                            "max_discount": max_discount or 20.0,
                            "personality_traits": personality_traits or [],
                            "negotiation_strategy": negotiation_strategy or [],
                            "exchanges": exchange_count,
                            "negotiated_price": negotiated_price,
                            "negotiated_discount": negotiated_discount,
                            "original_price": original_price,
                            "email_sent": "email_result" in locals() and email_result,
                            "calendar_scheduled": "meeting_result" in locals()
                            and meeting_result,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        print(
                            f"Deal information stored for conversation_id: {agent_session_id}"
                        )
                    except Exception as e:
                        import traceback

                        print(f"Error storing deal information: {e}")
                        print(traceback.format_exc())

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


@app.post("/chat/brewbot")
async def chat_with_brewbot(request: BrewBotChatRequest):
    """
    Send a message to BrewBot (external API agent) after agent-to-agent conversation.
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
            status_code=500, detail=f"Error sending message to BrewBot: {str(e)}"
        )


class TTSRequest(BaseModel):
    text: str


class DownloadReportRequest(BaseModel):
    conversation_id: str


def calculate_smart_savings_metrics(deal_data: Dict) -> Dict:
    """Calculate all SmartSavings metrics from deal data."""
    original_price = deal_data.get("original_price", 0)
    negotiated_price = deal_data.get("negotiated_price", 0)
    negotiated_discount = deal_data.get("negotiated_discount", 0)
    number_of_units = deal_data.get("number_of_units", 1)

    # Default assumptions
    profit_per_cup = 1.0  # $1 per cup
    cups_per_day = 500
    machine_lifetime_years = 5
    profit_per_cup_roi = 0.30  # $0.30 for ROI calculation

    # Calculate metrics
    total_original_price = original_price * number_of_units if original_price else 0
    total_negotiated_price = (
        negotiated_price * number_of_units if negotiated_price else 0
    )

    # 1. Absolute Savings
    absolute_savings = (
        total_original_price - total_negotiated_price
        if total_original_price and total_negotiated_price
        else 0
    )

    # 2. Final Discount (%)
    if total_original_price > 0:
        final_discount = (absolute_savings / total_original_price) * 100
    else:
        final_discount = negotiated_discount if negotiated_discount else 0

    # 3. Break-Even Time
    daily_profit = profit_per_cup * cups_per_day
    if daily_profit > 0 and total_negotiated_price > 0:
        break_even_days = total_negotiated_price / daily_profit
    else:
        break_even_days = 0

    # 4. ROI (1, 3, 5 years)
    annual_profit = cups_per_day * profit_per_cup_roi * 365
    roi_year_1 = (
        (annual_profit * 1) - total_negotiated_price if total_negotiated_price else 0
    )
    roi_year_3 = (
        (annual_profit * 3) - total_negotiated_price if total_negotiated_price else 0
    )
    roi_year_5 = (
        (annual_profit * 5) - total_negotiated_price if total_negotiated_price else 0
    )

    # 5. Cost Per Cup (Lifetime)
    total_cups = cups_per_day * 365 * machine_lifetime_years
    cost_per_cup = total_negotiated_price / total_cups if total_cups > 0 else 0

    # 6. Competitor Price Difference (using minPrice as competitor)
    competitor_price = None
    try:
        with open(project_root / "data" / "product.json", "r") as f:
            products = json.load(f)
            for product in products:
                if product.get("name") == deal_data.get("product_name"):
                    for version in product.get("versions", []):
                        competitor_price = version.get("minPrice", version.get("price"))
                        break
                    break
    except Exception:
        pass

    competitor_difference = 0
    percent_cheaper = 0
    if competitor_price and total_negotiated_price:
        competitor_total = competitor_price * number_of_units
        competitor_difference = competitor_total - total_negotiated_price
        if competitor_total > 0:
            percent_cheaper = (
                (competitor_total - total_negotiated_price) / competitor_total
            ) * 100

    # 7. Value-Added Perks (extract from conversation)
    perks = []
    conversation_text = " ".join(deal_data.get("conversation", []))
    perk_keywords = [
        "warranty",
        "installation",
        "training",
        "free",
        "included",
        "bonus",
        "extra",
    ]
    for keyword in perk_keywords:
        if keyword.lower() in conversation_text.lower():
            # Try to extract the perk context
            idx = conversation_text.lower().find(keyword.lower())
            if idx >= 0:
                snippet = conversation_text[max(0, idx - 50) : idx + 100]
                perks.append(snippet.strip())

    return {
        "absolute_savings": absolute_savings,
        "final_discount": final_discount,
        "break_even_days": break_even_days,
        "roi_year_1": roi_year_1,
        "roi_year_3": roi_year_3,
        "roi_year_5": roi_year_5,
        "cost_per_cup": cost_per_cup,
        "competitor_difference": competitor_difference,
        "percent_cheaper": percent_cheaper,
        "value_added_perks": perks[:3] if perks else ["Standard package included"],
        "total_original_price": total_original_price,
        "total_negotiated_price": total_negotiated_price,
        "number_of_units": number_of_units,
    }


@app.post("/download-report")
async def download_report(request: DownloadReportRequest):
    """Generate and download a PDF report for a completed deal."""
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="ReportLab library is not installed. Please install it with: pip install reportlab",
        )
    try:
        print(
            f"Download report requested for conversation_id: {request.conversation_id}"
        )
        print(
            f"Available conversation IDs in deal_storage: {list(deal_storage.keys())}"
        )

        if request.conversation_id not in deal_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Deal information not found for conversation_id: {request.conversation_id}. Available IDs: {list(deal_storage.keys())[:5]}",
            )

        deal_data = deal_storage[request.conversation_id]
        print(f"Deal data retrieved: {list(deal_data.keys())}")

        # Validate deal_data has required fields
        if not deal_data.get("conversation"):
            deal_data["conversation"] = ["No conversation history available"]

        metrics = calculate_smart_savings_metrics(deal_data)
        print(f"Metrics calculated successfully")

        # Use LLM to format the report content
        conversation_summary = (
            chr(10).join(deal_data.get("conversation", [])[:20])
            if deal_data.get("conversation")
            else "No conversation available"
        )

        report_prompt = f"""Create a professional SmartSavings Report for a coffee machine negotiation.

DEAL INFORMATION:
- Product: {deal_data.get("product_name", "N/A")}
- Units: {deal_data.get("number_of_units", "N/A")}
- Original Price: ${metrics.get("total_original_price", 0):,.2f}
- Negotiated Price: ${metrics.get("total_negotiated_price", 0):,.2f}
- Discount: {metrics.get("final_discount", 0):.2f}%

SMART SAVINGS METRICS:
1. Absolute Savings: ${metrics.get("absolute_savings", 0):,.2f}
2. Final Discount: {metrics.get("final_discount", 0):.2f}%
3. Break-Even Time: {metrics.get("break_even_days", 0):.1f} days
4. 1-Year ROI: ${metrics.get("roi_year_1", 0):,.2f}
5. 3-Year ROI: ${metrics.get("roi_year_3", 0):,.2f}
6. 5-Year ROI: ${metrics.get("roi_year_5", 0):,.2f}
7. Cost Per Cup: ${metrics.get("cost_per_cup", 0):.4f}
8. Competitor Savings: ${metrics.get("competitor_difference", 0):,.2f} ({metrics.get("percent_cheaper", 0):.2f}% cheaper)
9. Value-Added Perks: {", ".join(metrics.get("value_added_perks", []))}

CONVERSATION SUMMARY:
{conversation_summary}

Create a well-structured, professional report with:
- Executive Summary
- Detailed Metrics Section
- Conversation Highlights
- Recommendations

Format it as clear, professional text suitable for a PDF document."""

        # Get LLM-formatted report
        try:
            provider = "groq"
            selected_llm = "openai/gpt-oss-20b"
            formatted_report = await get_mcp_chatbot_response(
                report_prompt,
                f"{request.conversation_id}_report",
                provider,
                selected_llm,
                "mcp_chatbot",
            )
            print("LLM report generated successfully")
        except Exception as e:
            print(f"Error generating LLM report: {e}")
            # Fallback to a simple formatted report
            formatted_report = f"""Executive Summary:
This report summarizes the negotiation for {deal_data.get("product_name", "coffee machines")} with {deal_data.get("number_of_units", "N/A")} units.

Key Achievements:
- Total Savings: ${metrics.get("absolute_savings", 0):,.2f}
- Discount Achieved: {metrics.get("final_discount", 0):.2f}%
- Break-Even Period: {metrics.get("break_even_days", 0):.1f} days

Recommendations:
Based on the metrics, this negotiation has resulted in significant cost savings and a favorable return on investment."""

        # Generate PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch
        )
        story = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#0066cc"),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#0066cc"),
            spaceAfter=12,
        )

        # Title
        story.append(Paragraph("SmartSavings Negotiation Report", title_style))
        story.append(Spacer(1, 0.3 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(
            Paragraph(
                f"<b>Product:</b> {deal_data.get('product_name', 'N/A')}",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Units:</b> {deal_data.get('number_of_units', 'N/A')}",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Date:</b> {datetime.fromisoformat(deal_data.get('timestamp', datetime.now().isoformat())).strftime('%B %d, %Y')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Key Metrics Table
        story.append(Paragraph("Key Metrics", heading_style))
        metrics_data = [
            ["Metric", "Value"],
            ["Absolute Savings", f"${metrics.get('absolute_savings', 0):,.2f}"],
            ["Final Discount", f"{metrics.get('final_discount', 0):.2f}%"],
            ["Break-Even Time", f"{metrics.get('break_even_days', 0):.1f} days"],
            ["Cost Per Cup (Lifetime)", f"${metrics.get('cost_per_cup', 0):.4f}"],
            ["1-Year ROI", f"${metrics.get('roi_year_1', 0):,.2f}"],
            ["3-Year ROI", f"${metrics.get('roi_year_3', 0):,.2f}"],
            ["5-Year ROI", f"${metrics.get('roi_year_5', 0):,.2f}"],
        ]
        metrics_table = Table(metrics_data, colWidths=[3 * inch, 2 * inch])
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0066cc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(metrics_table)
        story.append(Spacer(1, 0.3 * inch))

        # Formatted Report Content
        story.append(Paragraph("Detailed Analysis", heading_style))
        for line in formatted_report.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))

        story.append(PageBreak())

        # Conversation Log
        story.append(Paragraph("Full Conversation Log", heading_style))
        for i, msg in enumerate(deal_data.get("conversation", []), 1):
            story.append(
                Paragraph(f"<b>Message {i}:</b> {msg[:500]}...", styles["Normal"])
            )
            story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        # Return PDF as download
        return StreamingResponse(
            BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=SmartSavings_Report_{request.conversation_id[:8]}.pdf"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating report: {str(e)}"
        )
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

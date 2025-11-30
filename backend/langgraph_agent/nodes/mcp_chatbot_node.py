import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage

# from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch

from langgraph.prebuilt import ToolNode

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configurations import AppConfiguration  # noqa: E402
from langgraph_agent.states.chatbotState import ChatbotState  # noqa: E402
from langgraph_agent.prompts import get_scout_system_prompt  # noqa: E402
from langgraph_agent.mcps.tool_loader import filter_tools_by_name  # noqa: E402
from langgraph_agent.tools.custom_tools import (
    get_all_gmail_tools,
    get_all_calendar_tools,
)  # noqa: E402
from langgraph_agent.generic.tool_runner import run_llm_tool_loop  # noqa: E402

load_dotenv()
settings = AppConfiguration()


class MCPChatbotNode:
    """
    MCP Chatbot node implementation with tool support.
    Supports MCP tools and other LangChain tools similar to graph.py and client.py.
    Handles tool calls by executing them and returning results.
    """

    def __init__(
        self,
        model,
        tools: Optional[List[BaseTool]] = None,
        custom_system_prompt: Optional[str] = None,
    ):
        """
        Initialize the chatbot node with an LLM and optional tools.

        Args:
            model: The language model to use
            tools: Optional list of tools to bind to the LLM (from MCP servers, Tavily, etc.)
            custom_system_prompt: Optional custom system prompt to use instead of default
        """
        self.llm = model
        # Initialize tools before loading additional ones to avoid attribute errors
        self.tools = list(tools) if tools else []
        self._load_additional_tools()
        print("Available Tools: ")
        for i in self.tools:
            print(i.name)
        print("--------------------------------")
        self.tool_node = ToolNode(self.tools) if self.tools else None
        self.custom_system_prompt = custom_system_prompt

        # Bind tools to LLM if provided (similar to graph.py)
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    def _load_additional_tools(self):
        """
        Load additional tools. This function adds Tavily and custom tools to the list of tools.
        Returns the list of tools.
        """

        # Add Tavily search tool if API key is available
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tools.append(TavilySearch(max_results=5, api_key=tavily_api_key))

        # Add Gmail and Calendar tools
        gmail_tools = get_all_gmail_tools()
        calendar_tools = get_all_calendar_tools()
        self.tools.extend(gmail_tools)
        self.tools.extend(calendar_tools)
        required_tool_names = settings.mcps.mcp_chatbot_node_config
        self.tools = filter_tools_by_name(self.tools, required_tool_names)

    async def process(self, state: ChatbotState) -> dict:
        """
        Processes the input state and generates a chatbot response.
        Returns the AI response as an AIMessage object to maintain conversation history.
        If tools are available, includes system prompt similar to graph.py.
        Handles tool calls by executing them and getting the final response.
        Supports multiple rounds of tool calls if needed.
        """
        # Create a copy of messages to avoid modifying the input state
        # self._load_additional_tools()
        messages = list(state["messages"])
        # Add system prompt if tools are available and not already present
        if self.tools:
            # Use custom prompt if provided, otherwise use the Scout system prompt
            if self.custom_system_prompt:
                system_prompt = self.custom_system_prompt
            else:
                system_prompt = get_scout_system_prompt()
            # Prepend system message if not already present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages

        return await run_llm_tool_loop(
            llm=self.llm,
            messages=messages,
            tools=self.tools,
            max_tool_iterations=10,
            session_key=state.get("session_key"),
        )


if __name__ == "__main__":
    import asyncio
    from langgraph_agent.llms.openai_llm import OpenAiLLM
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph_agent.mcps.tool_loader import load_tools_with_timeout

    async def main():
        """
        Initialize the MCP chatbot node and run the agent conversation loop.
        Similar to client.py but using the MCPChatbotNode directly.
        """
        # Create LLM instance
        user_controls_input = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "selected_llm": "gpt-4.1-mini",
        }
        llm = OpenAiLLM(user_controls_input)
        llm = llm.get_base_llm()

        # Load MCP tools
        tools = await load_tools_with_timeout()
        print(f"Tools loaded: {len(tools)}")

        # Create MCPChatbotNode instance with tools
        node = MCPChatbotNode(llm, tools=tools)

        # Default example
        user_input = "Can you subtract 67 from 99"
        print("\n ----  USER  ---- \n\n", user_input)
        print("\n ----  ASSISTANT  ---- \n\n")

        # Create state with user message
        state = {
            "messages": [
                SystemMessage(
                    content="You are a helpful and efficient assistant.",
                    additional_kwargs={},
                    response_metadata={},
                ),
                HumanMessage(
                    content=user_input, additional_kwargs={}, response_metadata={}
                ),
            ]
        }

        # Process with the node (now async)
        result = await node.process(state)

        # Extract and display the response
        result_messages = result.get("messages", [])
        if result_messages:
            last_message = result_messages[-1]
            # Extract content if it's a message object
            if hasattr(last_message, "content"):
                response_text = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                response_text = last_message["content"]
            else:
                response_text = str(last_message)

            # Print the response
            print(response_text)
        else:
            print("No response generated")

    # Run the async main function
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(main())

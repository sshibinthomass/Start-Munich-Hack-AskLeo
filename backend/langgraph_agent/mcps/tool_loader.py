"""
MCP tool loader with timeout handling.

This module provides functionality to load tools from MCP servers with
graceful timeout handling, allowing some servers to fail without blocking
the entire tool loading process.
"""

import asyncio
from typing import List, Sequence
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

from langgraph_agent.mcps.config import mcp_config

load_dotenv()


async def load_tools_with_timeout(
    server_timeout: float = 30.0,
) -> List[BaseTool]:
    """
    Load MCP tools from all configured servers with timeout handling.

    This function loads tools from each server individually with a timeout,
    preventing one slow server from blocking all tools. If a server times out
    or fails, it logs the error but continues loading tools from other servers.

    Args:
        server_timeout: Timeout in seconds for each server (default: 30.0)

    Returns:
        List of BaseTool objects from all successfully loaded servers
    """
    # Set up MCP client
    client = MultiServerMCPClient(connections=mcp_config["mcpServers"])

    # Load tools from each server individually with timeout handling
    # This prevents one slow server from blocking all tools
    tools = []

    async def load_tools_from_server(server_name: str):
        """Load tools from a single server with timeout handling"""
        try:
            server_tools = await asyncio.wait_for(
                client.get_tools(server_name=server_name), timeout=server_timeout
            )
            print(f"✓ Loaded {len(server_tools)} tools from '{server_name}'")
            for i in server_tools:
                print(i.name)
            return server_tools
        except asyncio.TimeoutError:
            print(f"⚠ Timeout loading tools from '{server_name}' (>{server_timeout}s)")
            return []
        except Exception as e:
            print(f"⚠ Error loading tools from '{server_name}': {e}")
            return []

    # Load tools from all servers in parallel, but allow individual failures
    if mcp_config["mcpServers"]:
        server_names = list(mcp_config["mcpServers"].keys())
        print(f"Loading tools from {len(server_names)} MCP servers...")

        # Use gather with return_exceptions=True to allow some servers to fail
        tool_results = await asyncio.gather(
            *[load_tools_from_server(name) for name in server_names],
            return_exceptions=True,
        )

        # Flatten the results and filter out exceptions
        for result in tool_results:
            if isinstance(result, list):
                tools.extend(result)
            elif isinstance(result, Exception):
                print(f"⚠ Exception loading tools: {result}")

    print(f"Loaded {len(tools)} tools total from MCP servers")
    return tools


def filter_tools_by_name(
    tools: Sequence[BaseTool], required_names: Sequence[str] | None
) -> List[BaseTool]:
    """
    Filter tools by the required names.

    Args:
        tools: Sequence of tool instances to filter.
        required_names: Iterable of tool names to keep.

    Returns:
        List of tools whose names match the required names.
    """
    if not required_names:
        return list(tools)

    required_set = {name for name in required_names if name}
    if not required_set:
        return list(tools)
    return [tool for tool in tools if tool.name in required_set]

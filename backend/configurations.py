from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    mcp_chatbot_node_config: list[str] = [
        "subtract",
        "multiply",
        "search",
        "fetch_content",
        "get_me",
        "search_repositories",
        "read_file",
        "read_text_file",
        "read_media_file",
        "read_multiple_files",
        "write_file",
        "edit_file",
        "create_directory",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "move_file",
        "search_files",
        "get_file_info"
    ]


class AppConfiguration(BaseSettings):
    mcps: MCPConfiguration = MCPConfiguration()

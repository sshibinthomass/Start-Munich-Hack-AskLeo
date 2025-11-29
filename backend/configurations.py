from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    mcp_chatbot_node_config: list[str] = [
        "product_get",
        "product_list_keys",
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
        "get_file_info",
        # Gmail tools
        "send_email",
        "list_emails",
        "read_email",
        "search_emails",
        "reply_to_email",
        # Calendar tools
        "create_event",
        "list_events",
        "delete_event",
    ]


class AppConfiguration(BaseSettings):
    mcps: MCPConfiguration = MCPConfiguration()

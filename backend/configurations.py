from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    mcp_chatbot_node_config: list[str] = [
        "product_get",
        "product_list_keys",
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

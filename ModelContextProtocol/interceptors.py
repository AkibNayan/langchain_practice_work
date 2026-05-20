import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent


async def append_structured_content(request: MCPToolCallRequest, handler):
    """Append structured content from artifacts to tool message."""
    result = await handler(request)
    if result.structuredContent:
        result.content += [
            TextContent(type="text", text=json.dumps(result.structuredContent))
        ]
    return result


client = MultiServerMCPClient({...}, tool_interceptors=append_structured_content)

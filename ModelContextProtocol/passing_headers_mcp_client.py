from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer Your token",
                "X-Custom-Header": "custom-value",
            },
        }
    }
)

tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-6", tools)

response = await agent.invoke({"messages": "what's the weather in nyc?"})

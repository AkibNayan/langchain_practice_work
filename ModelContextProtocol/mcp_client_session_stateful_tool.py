from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# create a session explicitly
async with client.session("server_name") as session:
    # Pass the session to load tools, resources or prompts
    tools = await load_mcp_tools(session)
    agent = create_agent("google_genai:gemini-3.1-pro-preview", tools)

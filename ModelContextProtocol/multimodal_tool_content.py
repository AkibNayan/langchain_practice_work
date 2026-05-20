from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})
tools = await client.get_tools()

agent = create_agent("claude-sonnet-4-6", tools)

result = agent.invoke({"messages": [{"role": "user", "content": "Take a screenshot of the current page."}]})

# Access multimodal content from tool message
for message in result["messages"]:
    if message.type == "tool":
        # Raw content in provider native format  
        print(f"Raw Content: {message.content}")
        
        # Standardized content blocks
        for block in message.content_blocks:
            if block["type"] == "text":
                print(f"Text: {block["text"]}")
            elif block["type"] == "image":
                print(f"Image url: {block.get("url")}")
                print(f"Base 64: {block.get("base64", "")[:50]}")
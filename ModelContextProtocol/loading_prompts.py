from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt

client = MultiServerMCPClient({...})
# Load a prompt by name
messages = await client.get_prompt("server_name", "summarizes")

# Load a prompt with arguments
messages = await client.get_prompt(
    "server_name", "code_review", arguments={"language": "python", "focus": "security"}
)

# Use the messages in your workflow
for message in messages:
    print(f"{message.type}: {message.content}")
    

async with client.session("server_name") as session:
    # load a prompt by name
    messages = await load_mcp_prompt(session, "summarizes")
    # Load a prompt with arguments
    messages = await load_mcp_prompt(
        session, "code_review", arguments={"language": "python", "focus": "security"}
    )

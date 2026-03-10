from langchain.chat_models import init_chat_model

model = init_chat_model(model="gpt-4.1-mini")


def get_weather(location: str) -> str:
    """Get the weather a location."""
    pass


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in San Francisco?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Arguments: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")

print(response.usage_metadata)

"""Streaming and Chunks"""
chunks = []
full_message = None
for chunk in model_with_tools.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk


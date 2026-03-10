from langchain.tools import tool
from langchain.chat_models import init_chat_model


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}."


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools("What's the weather like in Boston?")

for tool_call in response.tool_calls:
    # View tools calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Arguments: {tool_call['args']}")

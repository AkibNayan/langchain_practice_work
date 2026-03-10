from langchain.chat_models import init_chat_model
from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}."


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Boston and Tokyo?")

# The model may generate multiple tool calls
print(response.tool_calls)
"""
[
    {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
    {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'}
]
"""

# Execute all tools(can be done in parallel with async)
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    results.append(result)

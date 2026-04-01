from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what's the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(f"node: {metadata["langgraph_node"]}")
        print(f"content: {token.content_blocks}")
        print("\n")


"""
node: model
content: [{'type': 'tool_call_chunk', 'id': 'call_vbCyBcP8VuneUzyYlSBZZsVa', 'name': 'get_weather', 'args': '', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '{"', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': 'city', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '":"', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': 'San', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': ' Francisco', 'index': 0}]


node: model
content: [{'type': 'tool_call_chunk', 'id': None, 'name': None, 'args': '"}', 'index': 0}]


node: model
content: []


node: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]


node: model
content: []


node: model
content: [{'type': 'text', 'text': 'Here'}]


node: model
content: [{'type': 'text', 'text': ''s'}]


node: model
content: [{'type': 'text', 'text': ' what'}]


node: model
content: [{'type': 'text', 'text': ' I'}]


node: model
content: [{'type': 'text', 'text': ' got'}]


node: model
content: [{'type': 'text', 'text': ':'}]


node: model
content: [{'type': 'text', 'text': ' "'}]


node: model
content: [{'type': 'text', 'text': "It's"}]


node: model
content: [{'type': 'text', 'text': ' always'}]


node: model
content: [{'type': 'text', 'text': ' sunny'}]


node: model
content: [{'type': 'text', 'text': ' in'}]


node: model
content: [{'type': 'text', 'text': ' San'}]


node: model
content: [{'type': 'text', 'text': ' Francisco'}]


node: model
content: [{'type': 'text', 'text': '!"\n\n'}]
"""

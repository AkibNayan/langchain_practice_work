from langchain.chat_models import init_chat_model
from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}."


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_tools = model.bind_tools([get_weather])

for chunk in model_with_tools.stream("What's the weather in Boston and Tokyo?"):
    # Tool call chunks arrive progressively
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if id_ := tool_chunk.get("id"):
            print(f"Tool ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")


# Accumulate tool calls
gathered = None
for chunk in model_with_tools.stream("What's the weather in Boston and Tokyo?"):
    gathered = chunk if gathered is None else gathered+chunk
    print(gathered.tool_calls)

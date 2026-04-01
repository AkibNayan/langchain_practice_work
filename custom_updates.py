from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # Stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(model="claude-sonnet-4-6", tools=[get_weather])


for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in SF?"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])


"""
Looking up data for city: San Francisco
Acquired data for city: San Francisco
"""

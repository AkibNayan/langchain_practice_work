from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}."


agent = create_agent(
    "gpt-5",
    tools=[search],
    checkpointer=InMemorySaver()
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! my name is bob."}]},
    {"configurable": {"thread_id": 1}}
)

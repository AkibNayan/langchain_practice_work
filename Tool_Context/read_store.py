"""Read from store to access persisted user preferences."""

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


@tool
def get_preference(preference_key: str, runtime: ToolRuntime[Context]) -> str:
    """Get user preference from store."""
    user_id = runtime.context.user_id

    # Read from store: get existing user preference
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    if existing_prefs:
        value = existing_prefs.value.get(preference_key)
        return (
            f"{preference_key}: {value}"
            if value
            else f"No preference set for {preference_key}"
        )
    else:
        return "No preferences found"


agent = create_agent(
    model="gpt-4.5",
    tools=[get_preference],
    context_schema=Context,
    store=InMemoryStore(),
)

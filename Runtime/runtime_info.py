from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime


@dataclass
class Context:
    user_id: str


@tool
def fetch_user_email_preference(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user email preferences from the store."""
    user_id = runtime.context.user_id

    preferences: str = "The user prefers you to write a brief and polite info."

    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences

"""Read from state to check current session information."""

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if user is authenticated."""
    # Read from state: check current auth status
    current_state = runtime.state
    is_authenticated = current_state.get("is_authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"


agent = create_agent(model="gpt-4.5", tools=[check_authentication])

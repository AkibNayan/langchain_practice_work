"""Filter tools based on user permissions from runtime context."""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@dataclass
class Context:
    user_role: str


@wrap_model_call
def context_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on runtime context permissions."""
    # Read from Runtime Context: get user role
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # Admin get all tools
        pass
    elif user_role == "editor":
        # Editors can not delete
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # Viewer get read-only tools
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model="gpt-5.4",
    tools=[],  # [read_data, write_data, delete_data]
    middleware=[context_based_tools],
    context_schema=Context,
)

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


def select_relevant_tools():
    pass


@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Middleware to select relevant tools based on state/context."""
    # Select a small, relevant subset of tools based on state/context
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    
    return handler(request.override(tools=relevant_tools))


agent = create_agent(
    model="gpt-3.5-turbo",
    tools=[],
    middleware=[select_tools]
)
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable


def select_relevant_tool():
    pass


class ToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Middleware to select relevant tools based on state or context."""
        # Select a small, relevant subset of tools based on state/context.
        relevant_tools = select_relevant_tool(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))


agent = create_agent(model="gpt-4.1", tools=[], middleware=[ToolSelectorMiddleware()])

from langchain.agents.middleware import (
    before_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state["messages"])} messages")
    return None


@wrap_model_call
def retry_logic(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")


agent = create_agent(
    model="gpt-4.1", tools=[], middleware=[log_before_model, retry_logic]
)

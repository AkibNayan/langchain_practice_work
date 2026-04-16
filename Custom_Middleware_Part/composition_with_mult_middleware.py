from typing import Annotated, Callable
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import SystemMessage
from langgraph.types import Command
from typing_extensions import NotRequired


# Reducer Function
def _last_wins(a: str, b: str) -> str:
    """Reducer: last writer wins (outer overwrites inner.)"""
    return b


class CustomMiddlewareState(AgentState):
    """Agent state: trace_layer wins last wins (outer wins), messages use additive reducer."""

    # Non reducer field with last wins: both middleware write, outermost value wins.
    trace_layer: NotRequired[Annotated[str, _last_wins]]


class OuterMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ExtendedModelResponse:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "trace_layer": "outer",
                    "messages": [SystemMessage(content=["Outer ran"])],
                }
            ),
        )


class InnerMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ExtendedModelResponse:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "trace_layer": "inner",
                    "messages": [SystemMessage(content=["Inner ran"])],
                }
            ),
        )


"""
⚡ Simple Intuition
trace_layer → "Who wins?" → Outer wins
messages → "Keep everything?" → Yes, all messages kept
"""

"""Configure out format based on user preferences in store."""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str

class VerboseResponse(BaseModel):
    """Verbose response with details."""
    answer: str = Field(description="A detailed answer")
    sources: list[str] = Field(description="Sources used")


class ConciseResponse(BaseModel):
    """Concise response"""
    answer: str = Field(description="A brief answer")


@wrap_model_call
def store_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on Store preferences."""
    user_id = request.runtime.context.user_id
    
    # Read from user: get user's preferred response style
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)
    
    if user_prefs:
        style = user_prefs.value.get("response_style", "concise")
        if style == "verbose":
            request = request.override(response_format=VerboseResponse)
        else:
            request = request.override(response_format=ConciseResponse)

    return handler(request)


agent = create_agent(
    model="gpt-4.5",
    tools=[],
    middleware=[store_based_output],
    context_schema=Context,
    store=InMemoryStore()
)

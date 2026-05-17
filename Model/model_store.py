"""Use users prefered model from store"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


# Initialize available models once
MODEL_MAP = {
    "gpt-5.4": init_chat_model("gpt-5.4"),
    "gpt-5.4-mini": init_chat_model("gpt-5.4-mini"),
    "claude-sonnet-4-6": init_chat_model("claude-sonnet-4-6"),
}


@wrap_model_call
def store_based_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on store perferences."""
    user_id = request.runtime.context.user_id

    # Read from store: get users prefered model
    store = request.runtime.store
    user_prefs = store.get(("preferences"), user_id)

    if user_prefs:
        preferred_model = user_prefs.value.get("prefered_model")
        if preferred_model and preferred_model in MODEL_MAP:
            request = request.override(model=MODEL_MAP[preferred_model])

    return handler(request)


agent = create_agent(
    model="gpt-5.4",
    tools=[],
    middleware=[store_based_model],
    context_schema=Context,
    store=InMemoryStore(),
)

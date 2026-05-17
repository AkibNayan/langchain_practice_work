"""Use different model based on conversation length from state."""

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Initialize models once outside the middleware
large_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-5.4")
efficient_model = init_chat_model("gpt-5.4-mini")


@wrap_model_call
def state_based_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    if message_count > 20:
        # long conversation- use model with large context window
        model = large_model
    elif message_count > 10:
        # medium conversation
        model = standard_model
    else:
        # short conversation
        model = efficient_model

    request = request.override(model=model)

    return handler(request)


agent = create_agent(model="gpt-5.4", tools=[], middleware=[state_based_model])

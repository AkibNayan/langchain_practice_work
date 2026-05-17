"""Select model based on cost limit or environment from Runtime Context"""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable


@dataclass
class Context:
    cost_tier: str
    environment: str


# Initialize models once outside the middleware
premium_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-5.4")
budget_model = init_chat_model("gpt-5.4-mini")


@wrap_model_call
def context_based_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on runtime context"""
    # Read from runtime context: cost_tier and environment
    cost_tier = request.runtime.context.cost_tier
    environment = request.runtime.context.environment

    if environment == "production" and cost_tier == "premium":
        # Production premium users get best model
        model = premium_model
    elif cost_tier == "budget":
        # Budget tier get efficient model
        model = budget_model
    else:
        # All other users get standard model
        model = standard_model

    request = request.override(model=model)

    return handler(request)


agent = create_agent(
    model="gpt-5.4", tools=[], middleware=[context_based_model], context_schema=Context
)

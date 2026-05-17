"""configure structured output based on conversation state"""

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable


class SimpleResponse(BaseModel):
    """Simple response for early conversations."""

    answer: str = Field(description="A brief answer")


class DetailedResponse(BaseModel):
    """Detailed response for established conversations."""

    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")


@wrap_model_call
def state_based_output(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on State."""
    message_count = len(request.messages)

    if message_count < 3:
        # Early conversation, use simple format
        request = request.override(response_format=SimpleResponse)
    else:
        # Established conversation - use detailed format
        request = request.override(response_format=DetailedResponse)

    return handler(request)


agent = create_agent(model="gpt-4.5", tools=[], middleware=[state_based_output])

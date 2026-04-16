from collections.abc import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage


@wrap_model_call
def add_context(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    prev_system_msg = request.system_message.content_blocks
    new_content = list(prev_system_msg) + [
        {"type": "text", "text": "Additional context"}
    ]
    new_system_message = SystemMessage(content_blocks=new_content)
    return handler(request.override(system_message=new_system_message))

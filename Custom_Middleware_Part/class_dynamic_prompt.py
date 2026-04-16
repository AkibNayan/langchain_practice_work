from collections.abc import Callable
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage


class ContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        prev_system_msg = request.system_message.content_blocks
        new_content = list(prev_system_msg) + [
            {"type": "text", "text": "Additional context"}
        ]
        new_system_msg = SystemMessage(content_blocks=new_content)
        return handler(request.override(system_message=new_system_msg))

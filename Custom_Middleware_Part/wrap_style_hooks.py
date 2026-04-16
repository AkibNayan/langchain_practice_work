from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    for attempt in range(3):
        try:
            handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt+1}/3 after error: {e}")

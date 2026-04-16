from collections.abc import Callable
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command


class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"Executing tool: {request.tool_call["name"]}")
        print(f"Tool arguments: {request.tool_call['arguments']}")
        try:
            result = handler(request)
            print("Tool Completed Successfully")
            return result
        except Exception as e:
            print(f"Tool failed with error: {e}")
            raise

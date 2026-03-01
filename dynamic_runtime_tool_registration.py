from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest


# A tool that will be added dynamically at runtime
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill"""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools"""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        # This could be loaded from MCP server, database etc
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)


agent = create_agent(
    model="gpt-4.1",
    # tools=[get_weather], # only static tools registered here
    middleware=[DynamicToolMiddleware()],
)

result = agent.invoke(
    {"messages": [{"user": "role", "content": "Calculate a 20% tip on $85"}]}
)

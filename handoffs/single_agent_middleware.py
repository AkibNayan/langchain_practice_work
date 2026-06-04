from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define state with current_step tracker
class SupportState(AgentState):
    """Track which step is currently active."""

    current_step: str = "triage"
    warranty_status: str | None = None


# 2. Tools update current_step via Command
@tool
def record_warranty_status(
    status: str, runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record warranty status and transition to the next step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            # Transition to next step
            "current_step": "specialist",
        }
    )


# 3. Middleware applies dynamic configuration based on the current_step
@wrap_model_call
def apply_step_config(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    step = request.state.get("current_step", "triage")

    # Map steps to their configuration
    configs = {
        "triage": {
            "prompt": "Collect warranty information...",
            "tools": [record_warranty_status],
        },
        "specialist": {
            "prompt": "Provide solutions based on warranty: {warranty_status}",
            "tools": [...],  # [provide_solutions, escalate]
        },
    }

    config = configs[step]

    request = request.override(
        system_prompt=config["prompt"].format(**request.state), tools=config["tools"]
    )

    return handler(request)


# create agent with middleware
agent = create_agent(
    model=ChatGroq(model="llama-3.3-70b-versatile"),
    tools=[record_warranty_status],
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

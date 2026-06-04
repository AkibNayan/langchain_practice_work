from langchain.tools import tool
from langchain.messages import ToolMessage
from langgraph.types import Command


@tool
def transfer_to_specialist(runtime) -> Command:
    """Transfer to specialist agent"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Transfer to specialist",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "current_step": "specialist"  # Trigger to specialist agent
        }
    )
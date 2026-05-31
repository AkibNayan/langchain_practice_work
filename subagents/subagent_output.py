from typing import Annotated
from langchain.agents import AgentState
from langchain.tools import InjectedToolCallId
from langgraph.types import Command


@tool("subagent1_name", description="subagent1_description")
def call_subagent1(
    query: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    result = subagent1.invoke({"messages": [{"role": "user", "content": query}]})
    return Command(
        update={
            "example_state_key": result["example_state_key"],
            "messages": [
                ToolMessage(
                    content=result["messages"][-1].content, tool_call_id=tool_call_id
                )
            ],
        }
    )

from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    example_state_key: str


@tool("subagent1_name", description="subagent1_description")
def call_subagent1(query: str, runtime: ToolRuntime[None, CustomState]):
    # Apply any logic needed to transform the messages into a suitable input
    subagent_input = some_logic(query, runtime.state["messages"])
    result = subagent1.invoke(
        {
            "messages": subagent_input,
            # You can also pass other state keys here as needed.
            # Make sure to define these in both the main and subagent's state schemas.
            "example_state_key": runtime.state["example_state_key"],
        }
    )
    return result["messages"][-1].content

from langchain.tools import tool, ToolRuntime
from langgraph.prebuilt import ToolNode


@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Get the number of messages in the conversation."""
    messages = runtime.state["messages"]
    return f"There are {len(messages)} messages."


tool_node = ToolNode([get_message_count])

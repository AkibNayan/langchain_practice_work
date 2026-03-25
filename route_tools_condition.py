from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}."


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return str(eval(expression))


tools = ToolNode([search, calculator])


call_llm = ChatOpenAI(model="gpt-4.1")


builder = StateGraph(MessagesState)

builder.add_node("llm", call_llm)

builder.add_node("tools", ToolNode(tools))

# Define flow
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # Route to tools or END
builder.add_edge("tools", "llm")

graph = builder.compile()

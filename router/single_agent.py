from langgraph.types import Command


def classify_query(query: str) -> str:
    """Use the LLM to classify query and determine the appropriate agents."""
    ...


def route_query(state: State) -> Command:
    """Route to the appropriate agent based on the query classification."""
    active_agent = classify_query(state["query"])
    
    return Command(goto=active_agent)
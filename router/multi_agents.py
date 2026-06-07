from typing import TypedDict
from langgraph.types import Send


class ClassificationResult(TypedDict):
    query: str
    agent: str


def classify_query(query: str) -> list[ClassificationResult]:
    """Use the LLM to classify query and determine which agents to invoke."""
    ...


def route_query(state: State):
    """Route to relevant agents based on the query classification."""
    classification = classify_query(state["query"])

    return [Send(c["agent"], {"query": c["query"]}) for c in classification]

from langchain.messages import RemoveMessage
from langchain.agents import AgentState
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def delete_specific_messages(state: AgentState):
    messages = state["messages"]
    if len(messages) > 2:
        """remove the earliest two messages"""
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


def delete_all_messages(state: AgentState):
    """Remove all messages"""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

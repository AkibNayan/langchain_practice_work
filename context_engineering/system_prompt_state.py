"""Access Message count or conversation context from state."""

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base


agent = create_agent(model="gpt-3.5-turbo", tools=[], middleware=[state_aware_prompt])

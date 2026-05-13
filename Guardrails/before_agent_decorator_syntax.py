from typing import Any
from langchain.agents.middleware import before_agent, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.agents import create_agent

banned_keywords = ["hack", "exploit", "malware"]


@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    # Get the first user message
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    # check for banned keywords
    for keyword in banned_keywords:
        if keyword in content:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your requests.",
                    }
                ],
                "jump_to": "end",
            }
    return None


# Use the custom guardrail
agent = create_agent(
    model="gpt-4.1",
    tools=[],  # [search_tool, calculator_tool]
    middleware=[content_filter],
)

# This request will be blocked before any processing
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I hack into a database?"}]}
)

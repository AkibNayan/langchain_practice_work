from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.agents import create_agent


class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing Banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # check the banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I cannot process request containing inappropriate content. Please rephrase your request.",
                        }
                    ],
                    "jump_to": "end",
                }
        return None


# Use the custom guardrail

agent = create_agent(
    model="gpt-4.1",
    tools=[],  # [search_tool, calculator_tool]
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"])
    ],
)

# This result will be blocked before any processing
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I hack into a database?"}]}
)

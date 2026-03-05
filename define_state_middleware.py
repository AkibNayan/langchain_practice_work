from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    # tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        pass


agent = create_agent(
    model="gpt-4",
    # tools=tools,
    middleware=[CustomMiddleware()],
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "I prefer technical explanations."}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    }
)

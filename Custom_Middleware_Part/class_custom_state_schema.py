from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentMiddleware, AgentState
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    # Configure Middleware hook
    def before_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    # Configure Middleware Hook
    def after_model(
        self, state: CustomState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(model="gpt-4.1", tools=[], middleware=[CallCounterMiddleware()])

# Invoke with custom state
result = agent.invoke(
    {"messages": [HumanMessage("Hello")], "model_call_count": 0, "user_id": "user_123"}
)

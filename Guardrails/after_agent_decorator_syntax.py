from langchain.agents.middleware import AgentState, after_agent
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any
from langchain.agents import create_agent

safety_model = init_chat_model("gpt-5.4-mini")


@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model based guardrail: Use an LLM to evaluate response safety."""
    # Get the final AI response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use a model to evaluate safety
    safety_prompt = f"""Evaluate if this response is safe and appropriate.
    Respond with only 'SAFE' or 'UNSAFE'

    Response: {last_message.content}"""

    result = safety_model.invoke([{"role": "user", "content": safety_prompt}])

    if "UNSAFE" in result.content:
        last_message.content = (
            "I cannot provide that response. Please rephrase your request."
        )
    return None


# Use the safety guardrail in the agent
agent = create_agent(
    model="gpt-4.1",
    tools=[],  # [search_tool, calculator_tool]
    middleware=[safety_guardrail],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I make an explosives?"}]}
)

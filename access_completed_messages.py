from typing import Any, Literal
from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from pydantic import BaseModel


# structured data schema(model)
class ResponseSafety(BaseModel):
    """Evaluate a response as a safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]


safety_model = init_chat_model("openai:gpt-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtim: Runtime) -> dict[str, Any] | None:
    """Model based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    
    # get the model response
    if not state["messages"]:
        return None
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    
    # use another model to evaluate safety
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    
    result = model_with_tools.invoke([
        {
            "role": "system",
            "content": "Evaluate this AI response as generally safe or unsafe."
        },
        {
            "role": "user",
            "content": f"AI response: {last_message.text}"
        }
    ])
    
    stream_writer(result)
    
    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."
    return None
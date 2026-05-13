from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime


@before_model
def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
    """Block unauthenticated user when running on langgraph server."""
    server = runtime.server_info
    if server is not None and server.user is None:
        raise ValueError("Authentication required.")
    print(f"Thread: {runtime.execution_info.thread_id}")
    return None

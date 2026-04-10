from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model="gpt-4.1",
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end"
        )
    ]
)
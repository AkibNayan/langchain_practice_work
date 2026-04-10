from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            retry_delay=1,
            exit_behavior="end"
        )
    ]
)
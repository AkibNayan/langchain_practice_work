from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware


agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4.1-mini",
            "claude-3-5-sonnet-20241022"
        )
    ]
)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


# Single condition: trigger if tokens >= 4000
agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20 )
        )
    ]
)

# Multiple conditions: trigger if tokens >= 3000 or messages >= 6
agent2 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=[
                ("tokens", 3000),
                ("messages", 6)
            ],
            keep=("messages", 20)
        )
    ]
)

# Using fractional limits
agent3 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3)
        )
    ]
)
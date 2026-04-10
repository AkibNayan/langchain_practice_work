from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


agent = create_agent(
    model="gpt-5",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini", trigger=("tokens", 4000), keep=("messages", 20)
        )
    ],
)

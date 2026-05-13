from dataclasses import dataclass
from langchain.agents import create_agent


@dataclass
class Context:
    user_name: str


agent = create_agent(
    model="gpt-4.1", tools=[], context_schema=Context  # [search_tool, calculator_tool]
)

agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is my name?"}],
    },
    context=Context(user_name="John Smith"),
)

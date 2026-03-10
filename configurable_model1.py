from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(..., description="The city and state, e.g. San Fransisco, CA")


class GetPopulation(BaseModel):
    """Get the current population of a given location."""
    location: str = Field(..., description="The city and state, e.g. San Fransisco, CA")


model = init_chat_model(temperature=0)

model_with_tools = model.bind_tools([GetWeather, GetPopulation])

model_with_tools.invoke(
    "What's the bigger in 2024 LA or NYC",
    config={
        "configurable": {
            "model": "gpt-4.1-mini"
        }
    }
).tool_calls()

model_with_tools.invoke(
    "What's the bigger in 2024 LA or NYC",
    config={
        "configurable": {
            "model": "claude-haiku-4-5-20251001"
        }
    }
).tool_calls()

from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI


checkpointer = InMemorySaver()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in Dhaka?"}]}
)


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class Context:
    """Custom runtime context schema."""

    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on User ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

model = init_chat_model(
    "claude-sonnet-4-5-20250929", temperature=0.5, timeout=10, max_tokens=1000
)


# We use dataclass format but pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


agent1 = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy[ResponseFormat],
    checkpointer=checkpointer,
)

# thread id is a unique identifier for a given conversation
config = {"configurable": {"thread_id": "1"}}
response = agent1.invoke(
    {"messages": [{"role": "user", "content": "What's the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)
print(response["structured_response"])


response1 = agent1.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response1["structured_response"])
#################################
agent2 = create_agent("openai:gpt-5", tools=[get_weather_for_location, get_user_location])

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)
agent3 = create_agent(model=model, tools=[get_weather_for_location, get_user_location])
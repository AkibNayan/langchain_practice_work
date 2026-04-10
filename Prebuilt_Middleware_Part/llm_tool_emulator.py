from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import LLMToolEmulator


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"


# Emulate all tools
agent = create_agent(
    model="gpt-4.1", tools=[get_weather, send_email], middleware=[LLMToolEmulator()]
)

# Emulate specific tools only
agent2 = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(tools=[get_weather])],
)

# Use custom model for emulation
agent3 = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model="gpt-3.5-turbo")],
)

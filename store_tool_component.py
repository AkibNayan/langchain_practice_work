from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI


# Access Memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)

    return str(user_info.value) if user_info else "Unknow user"


# Update Memory
@tool
def save_user_info(
    user_id: str, user_info: dict[str, Any], runtime: ToolRuntime
) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users"), user_id, user_info)

    return "Successfully saved user info."


store = InMemoryStore()

model = ChatOpenAI(model="gpt-4.1")

agent = create_agent(model, tools=[get_user_info, save_user_info], store=store)


# First Session: Save User Info
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Save the following user: user_id: abc123, name: Foo, age: 25, email: foo@langchain.dev",
            }
        ]
    }
)


# Second Session: get user info
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Get user info for user with id 'abc123'"}
        ]
    }
)

# Output Data
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev

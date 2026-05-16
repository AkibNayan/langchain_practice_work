"""Access USER_ID or configuration from runtime context."""

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dataclass
class Context:
    user_role: str
    deployment_env: str


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # Read from runtime context: user role and environment
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read only operations."

    if env == "production":
        base += "\nBe extra careful with any data modifications."

    return base


agent = create_agent(
    model="gpt-3.5-turbo",
    tools=[],
    middleware=[context_aware_prompt],
    context_schema=Context,
)

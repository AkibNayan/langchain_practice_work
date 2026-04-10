from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from customize_agent_memory import search
from tool_call_limit_full_ex import database_tool


agent = create_agent(
    model="gpt-4.1",
    tools=[search, database_tool],
    middleware=[
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0)
    ],
)


agent2 = create_agent(
    model="gpt-4.1",
    tools=[search, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
            tools=["search"],
            retry_on=["ConnectionError", "TimeoutError"],
            on_failure="continue",
        )
    ],
)

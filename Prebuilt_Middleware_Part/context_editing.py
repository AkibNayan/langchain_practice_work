from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from customize_agent_memory import search
from tool_call_limit_full_ex import database_tool


agent = create_agent(
    model="gpt-4.1",
    tools=[search, database_tool],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=3,
                    clear_tool_inputs=False,
                    exclude_tools=[],
                    placeholder="[cleared]",
                )
            ]
        )
    ],
)

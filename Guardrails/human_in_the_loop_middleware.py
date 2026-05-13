from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-4.1",
    tools=[], # [search_tool, send_email_tool, delete_database_tool]
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval for sensitive operations
                "send_email": True,
                "delete_database": True,
                # Auto approve safe operations
                "search": False
            }
        )
    ],
    # Persist the state across interrupts
    checkpointer=InMemorySaver()
)

# Human in the loop require a thread id for persistance
config = {"configurable": {"thread_id": "user-1234"}}

# Agent will pause and wait for human approval before executing sensitive tools
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "send an email to the team"
    }]
}, config=config)

result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config # Same thread id to resume paused conversation
)

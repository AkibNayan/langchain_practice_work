from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-5.4",
    tools=[],  # [write_file, execute_sql, read_data]
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # All decisions (approve, edit, reject, respond) allowed
                "execute_sql": {
                    "allowed_decision": ["approve", "reject"]
                },  # no editing allowed
                "read_data": False,  # Safe operation, no approval needed
            },
            # Prefix for interrupt message - combined with tool name and args to form the full message
            # e.g.: "Tool execution pending approval: execute sql with query: 'DELETE FROM..'"
            # Individual tools can override this by specifying a "description" in their interrupt config
            description_prefix="Tool execution pending approval.",
        )
    ],
    # Human in the loop requires checkpointing to handle interrupts
    # In production use a persistent checkpointer like AsyncPostgresSaver
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "some_thread"}}

# stream agent progress and LLM tokens until interrupt
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Delete old records from database."}]},
    config = config,
    stream_mode = ["updates", "messages"],
    version="v2"
):
    if chunk["type"] == "messages":
        # LLM tokens
        token, metadata = chunk["data"]
        if token.content:
            print(token.content, end="", flush=True)
    elif chunk["type"] == "updates":
        # check for interrupts
        if "__interrupt__" in chunk["data"]:
            print(f"\n\nInterrupt: {chunk['data']['__interrupt__']}")


# Resume with stream after human decision  
for chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode=["updates", "messages"],
    version="v2"
):
    if chunk["type"] == "messages":
        # LLM tokens
        token, metadata = chunk["data"]
        if token.content:
            print(token.content, end="", flush=True)


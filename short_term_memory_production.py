from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}."


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # auto create tables in PostgreSQL
    agent = create_agent(
        "gpt-5",
        tools=[search],
        checkpointer=checkpointer
    )

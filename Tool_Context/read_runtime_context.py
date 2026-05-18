from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent


@dataclass
class Context:
    user_id: str
    api_key: str
    db_connection: str


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch data using runtime context configuration."""
    # Read from runtime context: get API_key and db connection
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    db_connection = runtime.context.db_connection

    # use configuration to fetch data
    results = perform_database_query(db_connection, query, api_key)

    return f"Found {len(results)} results for user {user_id}"


agent = create_agent(model="gpt-4.5", tools=[fetch_user_data], context_schema=Context)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Get my data"}]},
    context=Context(user_id="123", api_key="sk...", db_connection="postgresql://..."),
)

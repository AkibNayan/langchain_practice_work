from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware


agent = create_agent(
    model="gpt-4.1",
    # tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)

from langchain.tools import tool, ToolRuntime


@tool
def context_aware_tool(runtime: ToolRuntime) -> str:
    """A tool that uses execution and server info."""
    # Access thread and run ID
    info = runtime.execution_info
    print(f"Thread ID: {info.thread_id}, Run ID: {info.run_id}")

    # Access server info (only available in langgraph server)
    server = runtime.server_info
    if server is not None:
        print(f"Assistant: {server.assistant_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")

    return "Done"

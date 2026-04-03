from access_completed_messages1 import (
    agent,
    _render_message_chunk,
    _render_completed_message,
)
from langchain.messages import AIMessageChunk


input_message = {"role": "user", "content": "what's the weather in Boston?"}
full_message = None

for chunk in agent.stream(
    {"messages": [input_message]}, stream_mode=["messages", "updates"], version="v2"
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
            full_message = token if full_message is None else full_message + token
            if token.chunk_position == "last":
                if full_message.tool_calls:
                    print(f"Tool calls: {full_message.tool_calls}")
                full_message = None
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "tools":
                _render_completed_message(update["messages"][-1])

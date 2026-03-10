from langchain.messages import HumanMessage, AIMessage

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider native format
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.png"}
])

# Anthropic standard content blocks
messages1 = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "Wzafl.."},
        {"type": "text", "text": "..."}
    ],
    response_metadata={"model_provider": "anthropic"}
)
print(messages1.content_blocks)
# OpenAI standard content blocks
messages2 = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"}
            ]
        },
        {"type": "text", "text": "...", "id": "msg_abc123"}
    ],
    response_metadata={"model_provider": "openai"}
)
print(messages2.content_blocks)




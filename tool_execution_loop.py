from langchain.chat_models import init_chat_model
from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}."


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)
# Bind (Potentially multiple) tools to the model
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Pass results back to the model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.text)

# The current weather in Boston is 72 F and sunny.

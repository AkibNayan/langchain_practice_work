from langchain.messages import AIMessage, HumanMessage
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    # tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        pass


agent = create_agent(
    model="gpt-4",
    # tools=tools,
    middleware=[CustomMiddleware()],
)


for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings."}]}, stream_mode="values"):
    # Each chunk contain the full state at that point
    latest_message = chunk['messages'][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
        
        
        
    





conversation = [
    {"role": "system", "content": "You are a helpful assistant that translate English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J`adore la programmation."}
    {"role": "user", "content": "Translate: I love building application."}
]

response = agent.invoke(conversation)
print(response)

# MessageObjects
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translate English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J`adore la programmation."),
    HumanMessage("Translate: I love building application.")
]

response = agent.invoke(conversation)
print(response)

# Basic Text Streaming
for chunk in agent.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
    

for chunk in agent.stream("What color is the sky?"):
    for block in chunk.content_blocks:
        if block['type'] == 'reasoning' and (reasoning := block.get('reasoning')):
            print(f"Reasoning: {reasoning}")
        elif block['type'] == 'tool_call_chunk':
            print(f"Tool call chunk: {block}")
        elif block['type'] == 'text':
            print(block['text'])
        else:
            pass
        
# Streaming events
async for event in agent.astream_events("Hello"):
    if event['event'] == 'on_chat_model_start':
        print(f"Input: {event['data']['input']}")
    elif event['event'] == 'on_chat_model_stream':
        print(f"Token: {event['data']['chunk'].text}")
    elif event['event'] == 'on_chat_model_end':
        print(f"Full message: {event['data']['output'].text}")
    else:
        pass
    

# Batch
responses = agent.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])            
for response in responses:
    print(response)

for response in agent.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
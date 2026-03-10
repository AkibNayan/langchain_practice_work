from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage


model = init_chat_model(model="gpt-4.1-mini")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

messages = [system_msg, human_msg]

response = model.invoke(messages)  # Return AI Message

messages1 = [
    SystemMessage("You are a poetry expert."),
    HumanMessage("Write a poem about spring."),
    AIMessage("Cherry blossoms bloom..."),
]
response1 = model.invoke(messages1)  # Return AI Message

# Dictionany format also works
messages2 = [
    {"role": "system", "content": "You are a poetry expert."},
    {"role": "user", "content": "Write a poem about spring."},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]

response2 = model.invoke(messages2)  # Return AI Message

"""System Messages"""
system_msg3 = SystemMessage("You are a helpful coding assistant.")
messages3 = [
    system_msg3,
    HumanMessage("How do I create a Rest API?")
]
response3 = model.invoke(messages3)

# Detailed persona
system_msg4 = SystemMessage(
    """
    You are a senior python developer with expertise in web frameworks.
    Always provide code examples and explain your reasoning.
    Be concise but thorough in your explanations.
    """
)
message4 = [
    system_msg4,
    HumanMessage("How do I create a REST API")
]
response4 = model.invoke(message4)

"""AI Message"""
# Create an AI Message manually
ai_msg5 = AIMessage("I'd be happy to help you with that question!")
message5 = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can I help you?"),
    ai_msg5,
    HumanMessage("Great! what is 2+2?")
]

response5 = model.invoke(message5)

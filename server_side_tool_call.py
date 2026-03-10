from langchain.chat_models import init_chat_model

model = init_chat_model(model='gpt-4.1')

tool = {"type": "web_search"}

model_with_tools = model.bind_tools([tool])
response = model_with_tools.invoke("What was a positive news story today?")

print(response.content_blocks)
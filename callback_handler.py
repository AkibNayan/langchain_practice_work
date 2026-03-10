from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler


model1 = init_chat_model(model="gpt-4.1-mini")
model2 = init_chat_model(model="claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()

result1 = model1.invoke("Hello", config={"callbacks": [callback]})
result2 = model2.invoke("Hello", config={"callbacks": [callback]})

print(callback.usage_metadata)

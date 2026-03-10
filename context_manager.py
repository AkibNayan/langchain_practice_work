from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

model1 = init_chat_model(model="gpt-4.1-mini")
model2 = init_chat_model(model="claude-haiku-4-5-20251001")

with get_usage_metadata_callback() as cb:
    model1.invoke("Hello")
    model2.invoke("Hello")
    print(cb.usage_metadata)

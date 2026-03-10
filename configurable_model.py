from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    'What is your name?',
    config={
        "configurable": {"model": "gpt-4.1-mini"}
    }
)

configurable_model.invoke(
    'What is your name?',
    config={
        "configurable": {"model": "claude-haiku-4-5-20251001"}
    }
)

# Configurable models with default values
first_model = init_chat_model(
    model="gpt-4.1-mini",
    temperature=0.9,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
    config_prefix="first" # Useful when you have a chain with multiple models
)
first_model.invoke("What is your name?")

first_model.invoke(
    "What is your name?",
    config={
        "configurable": {
            'first_model': "claude-sonnet-4-6",
            'first_temperature': 0.5,
            'first_max_tokens': 100
        }
    }
)

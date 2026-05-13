from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[], # [customer_service_tool, email_tool],
    middleware=[
        # REDACT Email in user input before sending to model
        PIIMiddleware(
           "email",
           strategy="redact",
           apply_to_input=True 
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            'credit_card',
            strategy="mask",
            apply_to_input=True
        ),
        # Block API keys- raise error if detected in user input
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True
        )
    ]
)

# When the user provides PII, it will be handled according to the strategy
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100",
            }
        ]
    }
)

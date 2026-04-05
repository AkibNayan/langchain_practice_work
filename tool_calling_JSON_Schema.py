from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


product_review_schema = {
    "type": "object",
    "description": "Analysis of a product review.",
    "properties": {
        "rating": {
            "type": ["integer", "null"],
            "description": "The rating of the product (1-5)",
            "minimum": 1,
            "maximum": 5,
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"],
            "description": "The sentiment of the review",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The key points of the reivew.",
        },
    },
    "required": ["sentiment", "key_points"],
}


agent = create_agent(model="gpt-5", response_format=ToolStrategy(product_review_schema))


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
            }
        ]
    }
)
result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}

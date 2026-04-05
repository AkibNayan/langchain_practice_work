from dataclasses import dataclass
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


@dataclass
class ProductReview:
    """Analysis of a product review."""

    rating: int | None  # The rating of the product 1-5
    sentiment: Literal["positive", "negative"]  # The sentiment of the review
    key_points: list[str]  # The key points of the review


agent = create_agent(model="gpt-5", response_format=ToolStrategy(ProductReview))

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze the review: 'Great products: 5 out of 5 start. Fast shipping, but expensive.",
            }
        ]
    }
)

print(result["structured_response"])

# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}

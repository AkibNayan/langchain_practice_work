from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating of the product, ge=1, le=5")
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    key_points: list[str] = Field(
        description="Key points of the review, lowercase, 1-3 words each"
    )


class CustomerComplaint(BaseModel):
    """A customer complaint about a product or service."""

    issue_type: Literal["product", "service", "shipping", "billing"] = Field(
        description="The type of issue"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="The severity of the complaint"
    )
    description: str = Field(description="Brief description of the complaint.")


agent = create_agent(
    model="gpt-5", response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze this review: 'Great products: 5 out of 5 stars. Fast shipping, but expensive.",
            }
        ]
    }
)

print(result["structured_response"])

# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])

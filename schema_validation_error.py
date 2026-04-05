from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5. ge=1, le=5")
    comment: str = Field(description="Review comment")


agent = create_agent(
    model="gpt-5",
    response_format=ToolStrategy(ProductRating),  # Default handle_erros=True
    system_prompt="You are a helpful assistant that parses product reviews. Don't make any field or value type",
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]}
)


"""
================================ Human Message =================================

Parse this: Amazing product, 10/10!
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_1)
 Call ID: call_1
  Args:
    rating: 10
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Error: Failed to parse structured output for tool 'ProductRating': 1 validation error for ProductRating.rating
  Input should be less than or equal to 5 [type=less_than_equal, input_value=10, input_type=int].
 Please fix your mistakes.
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_2)
 Call ID: call_2
  Args:
    rating: 5
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Returning structured response: {'rating': 5, 'comment': 'Amazing product'}
"""

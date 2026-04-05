from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError
from multiple_struc_output_error import ContactInfo, EventDetails
from langchain.agents import create_agent
from typing import Union
from langchain.agents.structured_output import ToolStrategy


def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with that format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails], handle_errors=custom_error_handler
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th",
            }
        ]
    }
)


for msg in result["messages"]:
    # If message is actually a ToolMessage object (not a dict), check its class name
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)
    # If message is a dictionary or you want a fallback
    elif isinstance(msg, dict) and msg.get("tool_call_id"):
        print(msg["content"])

"""
================================= Tool Message =================================
Name: ToolStrategy

There was an issue with the format. Try again.

================================= Tool Message =================================
Name: ToolStrategy

Multiple structured outputs were returned. Pick the most relevant one.

================================= Tool Message =================================
Name: ToolStrategy

Error: <error message>
"""

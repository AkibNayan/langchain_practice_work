from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""

    task: str = Field(description="The specific task to be completed.")
    assignee: str = Field(description="Person responsible for the task.")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


agent = create_agent(
    model="gpt-5",
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to the meeting notes!",
    ),
)

agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "From our meeting: Sarah needs to update the project timeline as soon as possible.",
            }
        ]
    }
)

"""
================================ Human Message =================================

From our meeting: Sarah needs to update the project timeline as soon as possible
================================== Ai Message ==================================
Tool Calls:
  MeetingAction (call_1)
 Call ID: call_1
  Args:
    task: Update the project timeline
    assignee: Sarah
    priority: high
================================= Tool Message =================================
Name: MeetingAction

Action item captured and added to meeting notes!
"""

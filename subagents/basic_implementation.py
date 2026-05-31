from langchain.tools import tool
from langchain.agents import create_agent

# Create a subagent
subagent = create_agent(model="google_genai:gemini-3.5-flash", tools=[...])


# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# Main agent with subagent as a tool
agent = create_agent(model="google_genai:gemini-3.5-flash", tools=[call_research_agent])

from langchain.tools import tool
from langchain.agents import create_agent

# Sub-agents developed by different teams
research_agent = create_agent(
    model="gpt-5.6",
    system_prompt="You are a research specialist..."
)

writer_agent = create_agent(
    model="gpt-5.6",
    system_prompt="You are a writing specialist..."
)

# Registry of available sub-agents  
SUBAGENTS = {
    "research": research_agent,
    "writer": writer_agent
}

@tool
def task(
    agent_name: str,
    description: str
) -> str:
    """Launch an ephemeral sub-agent for a task
    
    Available agent:
    - research: Research and fact-finding 
    - writer: Content creation and editing 
    """
    agent = SUBAGENTS[agent_name]
    result = agent.invoke(
        {"messages": [{"role": "user", "content": description}]}
    )
    return result["messages"][-1].content


# Main coordinator agent
main_agent = create_agent(
    model="gpt-5.6",
    tools=[task],
    system_prompt=(
        "You coordinate specialized subagents."
        "Available: research(fact-finding), "
        "writer(content creation)"
        "Use the task tool to delegate work."
    )
)
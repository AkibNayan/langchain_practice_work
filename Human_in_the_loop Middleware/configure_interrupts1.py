from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


agent = create_agent(
    model="gpt-5.4",
    tools=[],  # [write_file, execute_sql, read_data]
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                "read_data": False,
            },
            description_prefix="Tool execution pending approval.",
        )
    ],
    checkpointer=InMemorySaver(),
)

# Human in the loop leverages langGraph's persistence layer.
# You must provide a thread ID to associate the execution with a conversation thread.
# so the conversation can be paused and resumed (as is needed for human review).

config = {"configurable": {"thread_id": "some_thread"}}
# Run the graph until the interrupt is hit
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Delete old records from database."}]
    },
    config=config,
    version="v2"
)

# result is a graphOutput with .value and .interrupts
print(result.interrupts)
# > (
# >    Interrupt(
# >       value={
# >          'action_requests': [
# >             {
# >                'name': 'execute_sql',
# >                'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
# >                'description': 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
# >             }
# >          ],
# >          'review_configs': [
# >             {
# >                'action_name': 'execute_sql',
# >                'allowed_decisions': ['approve', 'reject']
# >             }
# >          ]
# >       }
# >    ),
# > )

# Resume with approval decision
agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config,
    version="v2"
)


agent.invoke(
    Command(
       # Decisions are provided as a list. one per action under review.
       # The order of decisions must match the order of action
       # in the interrupt request
       resume = {
           "decisions": [
               {
                   "type": "approve"
               }
           ]
       } 
    ),
    config=config,  # Same thread ID to resume the paused conversation
    version="v2"
)

agent.invoke(
    Command(
        resume = {
            "decisions": [
                {
                    "type": "edit",
                    # Edited action with tool name and arguments
                    "edited_action": {
                        "name": "new_tool_name",
                        "args": {"key_1": "new_value", "key_2": "original_value"}
                    }
                }
            ]
        }
    ),
    config=config,
    version="v2"
)

agent.invoke(
    Command(
        resume = [
            {
                "type": "reject",
                # An explanation of why the action was rejected
                "message": "No, this is wrong because ..., instead do this ..."
            }
        ]
    )
    config=config,
    version="v2"
)

agent.invoke(
    Command(
        resume = [
            {
                "type": "respond",
                # The humans reply, returned directly as the tool result
                "message": "Blue"
            }
        ]
    ),
    config=config,
    version="v2"
)



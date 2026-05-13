from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from after_agent_class_syntax import SafetyGuardrailMiddleware
from before_agent_class_syntax import ContentFilterMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],  # [search_tool, calculator_tool]
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),
        # Layer 2: PII middleware (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        # Layer 4: Model based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)

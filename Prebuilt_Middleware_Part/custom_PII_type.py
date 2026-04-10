from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
import re


# Method-1: Regex Pattern String
agent1 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block")
    ],
)

# Method-2: Compiled Regex Pattern
agent2 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=re.compile(r"\+?\d{1, 3}[\s.-]?\d{3, 4}[\s.-]?\d{4}"),
            strategy="mask",
        )
    ],
)


# Method-3: Custom Detector Function
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """Detect US SSN with validation.

    Returns a list of dictionaries with 'text', 'start', and 'end' keys.
    """
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"

    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # validate: first three digit should not be 000, 666, 900-999
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({"text": ssn, "start": match.start(), "end": match.end()})
    return matches


agent3 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[PIIMiddleware("ssn", detector=detect_ssn, strategy="hash")],
)

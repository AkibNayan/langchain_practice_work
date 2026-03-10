from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated


class Actor(TypedDict):
    name: str
    role: str


class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: Annotated[float | None, ..., "Budget in millions USD"]


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_structure = model.with_structured_output(MovieDetails, include_raw=True)

response = model_with_structure.invoke("Provide details about the movie Inception.")
print(response)


# Multimodal
response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [
#     {"type": "text", "text": "Here's a picture of a cat"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]

"""Stream Reasoning output"""
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)


"""Complete Reasoning output"""
response = model.invoke("Why do parrots have colorful feathers?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))

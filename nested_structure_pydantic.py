from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


# Schemas can be nested:
class Actor(BaseModel):
    name: str
    role: str


class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_structure = model.with_structured_output(MovieDetails, include_raw=True)

response = model_with_structure.invoke("Provide details about the movie Inception.")
print(response)

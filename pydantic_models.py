from pydantic import Field, BaseModel
from langchain.chat_models import init_chat_model


class Movie(BaseModel):
    """A movie with details"""

    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie rating out of 10")


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_structure = model.with_structured_output(Movie, include_raw=True)

response = model_with_structure.invoke("Provide details about the movie Inception.")

print(response) # Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)

# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }

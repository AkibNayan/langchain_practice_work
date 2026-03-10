from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated


class MovieDict(TypedDict):
    """A movie with details"""

    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released."]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]


model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("Provide details about the movie Inception.")
print(
    response
)  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}

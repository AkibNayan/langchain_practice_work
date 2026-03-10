import json
from langchain.chat_models import init_chat_model


json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the movie"},
        "year": {"type": "integer", "description": "The year the movie was released"},
        "director": {"type": "string", "description": "The director of the movie"},
        "rating": {"type": "float", "description": "The movie's rating out of 10"},
    },
    "required": ["title", "year", "director", "rating"],
}

model = init_chat_model(model="gpt-4.1", temperature=0.7, max_tokens=1000)

model_with_structure = model.with_structured_output(json_schema, method="json_schema")

response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, ...}

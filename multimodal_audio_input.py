# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {
            "type": "audio",
            "base64": "JVBERi0xLjQKJcfsj6IKNSAwIG9iago",
            "mime_type": "audio/wav",
        },
    ],
}

# From provider managed file ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {"type": "audio", "file_id": "file-abc123"},
    ],
}

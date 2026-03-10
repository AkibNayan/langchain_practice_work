# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this PDF document."},
        {"type": "file", "url": "https://example.com/path/to/document.pdf"}
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {
            "type": "file",
            "base64": "JVBERi0xLjQKJcfsj6IKNSAwIG9iago",
            "mime_type": "application/pdf"
        }
    ]
}

# From provider managed file ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {"type": "file", "file_id": "file-abc123"}
    ]
}
# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {
            "type": "video",
            "base64": "JVBERi0xLjQKJcfsj6IKNSAwIG9iago",
            "mime_type": "video/mp4",
        },
    ],
}

# From provider managed file ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {"type": "video", "file_id": "file-abc123"},
    ],
}

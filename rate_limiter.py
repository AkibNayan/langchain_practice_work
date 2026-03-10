from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10 seconds
    check_every_n_seconds=0.1,  # check every 100ms whether allowed to make a request
    max_bucket_size=10  # controls the maximum burst size
)

model = init_chat_model(
    model="gpt-5",
    model_provider="openai",
    rate_limiter=rate_limiter
)

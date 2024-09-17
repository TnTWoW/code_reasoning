import os
import asyncio
from openai import AsyncOpenAI
from openai import OpenAIError

# Set up the proxy (if required)
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# Initialize the OpenAI client
client = AsyncOpenAI(
    api_key='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',
)

async def main() -> None:
    # Create a chat completion request
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )

    # Print the response content
    print(chat_completion)

# Run the async main function
asyncio.run(main())

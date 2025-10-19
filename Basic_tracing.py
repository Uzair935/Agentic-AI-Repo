from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel,trace,set_tracing_disabled
import asyncio
import os
from dotenv import load_dotenv

set_tracing_disabled(False)

external_client = AsyncOpenAI(
    api_key = os.getenv("gemini_api_key"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/"
)

llm_model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client= external_client
)

async def main():
    agent = Agent(
        name = "Joker",
         instructions="You're job is tell jokes",
         model = llm_model
    )

    with trace("Joke Workflow"):
        result = await Runner.run(agent,"Tell me a joke")
        print(result.final_output)

asyncio.run(main())
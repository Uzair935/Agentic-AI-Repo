from agents import Agent, Runner,AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings, RunContextWrapper, ItemHelpers 
import asyncio
import datetime
import random
import os
from dotenv import load_dotenv

set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key = os.getenv("gemini_api_key"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/"
)
llm_model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client = external_client
)

@function_tool
def how_many_jokes():
    """This gives how many jokes to tell"""
    return random.randint(2, 5)

async def main():
    agent = Agent(
        name = "Joker",
        instructions="You are a helpful assistant. First, determine how many jokes to tell, then provide jokes.",
        tools=[how_many_jokes],
        model = llm_model
    )

    result = await Runner.run_streamed(agent, input= "Hello, tell me some jokes")

    async for event in result.stream_events():
        if event.item.type == "tool_call_output_item":
            print(f"Tool output : {event.item.output}") 
        elif event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))

asyncio.run(main()) 
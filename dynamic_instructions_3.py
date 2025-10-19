from agents import Agent, Runner,AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings, RunContextWrapper 
import asyncio
import datetime
import os
from dotenv import load_dotenv

set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key = os.getenv("gemini_api_key"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/"
)

llm_model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client= external_client
)

async def instruc(context: RunContextWrapper,agent: Agent):
    await asyncio.sleep(0.1)
    current_time = datetime.datetime.now()

    return f"""You are {agent.name}, an AI assistant with real-time capabilities.
    Current time: {current_time.strftime('%H:%M')}
    Provide helpful and timely responses."""

agent = Agent(
    name = "Async Assistant",
    instructions = instruc,
    model= llm_model
)

runner = Runner()
result = runner.run_sync(starting_agent=agent, input="What is the current time?")
print(result.final_output)
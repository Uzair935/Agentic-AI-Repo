from agents import Agent,Runner,OpenAIChatCompletionsModel,AsyncOpenAI,set_tracing_disabled,function_tool
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

@function_tool
def multiply(a:int,b:int):
    """🧮 Exact multiplication (use this instead of guessing math)."""
    return a*b

@function_tool
def addition(a:int,b:int):
    """➕  Exact addition (use this instead of guessing math)."""
    return a+b


agent = Agent(name = "Assistant",
        instructions = "You are a helpful agent.Always use tools for math question where possible. Use the DMAS rule. Explain answers briefly and shortly for beginners.",
        model = llm_model,
        tools = [multiply,addition])

result = Runner.run_sync(agent,"Solve: 19 + 23 * 2?")
print(result.final_output)
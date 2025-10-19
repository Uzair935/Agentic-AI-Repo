from agents import Agent, Runner,AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings, RunContextWrapper 
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

def dynamic_instructions(context: RunContextWrapper,agent: Agent):
    return f"You are {agent.name}. Adapt to the user's needs"

agent = Agent(
    name = "Smart Assistant",
    model= llm_model,
    instructions = dynamic_instructions
)

runner = Runner()
result = runner.run_sync(starting_agent=agent, input="What's the capital of France?")
print(result.final_output)
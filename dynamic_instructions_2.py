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

def context_aware(context: RunContextWrapper,agent: Agent):
    message_count = len(getattr(context,'messages',[])) # getattr(object, attribute_name, default_value)
    if message_count == 0:
        return f"You are a Welcoming assistant. Greet the user with a friendly message"
    elif message_count < 3:
        return "You are a Helpful assistant. Be encouraging and detailed."
    else:
        return "You are a Experienced assistant. Be concise and through."

agent = Agent(
    name = "Context Aware Assistant",
    instructions = context_aware,
    model= llm_model
)

runner = Runner()
result = runner.run_sync(starting_agent=agent, input="What is National anthem of USA?")
print(result.final_output)
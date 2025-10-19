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


# class StatefulInstructions:
#     def __init__(self):
#         self.interaction_counter = 0
    
#     def __call__(self, context : RunContextWrapper, agent : Agent):
#         self.interaction_counter += 1
#         return self.interaction_counter

# message_count = StatefulInstructions()

def exploring_context(context: RunContextWrapper,agent: Agent):
    if context.context is None:
        context.context = {}
    
    if "interaction_counter" not in context.context:
        context.context["interaction_counter"] = 0

    context.context["interaction_counter"] += 1

    message_counter = context.context["interaction_counter"]

    user_name = context.context.get("name", "User")
    return f"You are {agent.name}. Talking to {user_name}. Message #{message_counter}." 

llm_model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client= external_client
)

agent = Agent(
    name = "Async Assistant",
    instructions = exploring_context,
    model= llm_model
)

runner = Runner()
result = runner.run_sync(starting_agent=agent, input="How many messages had took place between you and me now?")    
print(result.final_output)
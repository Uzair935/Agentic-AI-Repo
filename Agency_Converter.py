from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,set_tracing_disabled,function_tool,ModelSettings
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
def Currency_Changer(amount:int):
    """This changes USD to Pkr"""
    USD_to_Pkr = amount*280
    return(f"${amount} is Pkr{USD_to_Pkr}")

Currency_Agent = Agent(name = "Currency_Changer",
instructions = "You are a currency changer. You convert USD to Pkr.Use tools provided",
model = llm_model,
tools = [Currency_Changer],
model_settings = ModelSettings(temperature = 0.2,max_token = 100))

result = Runner.run_sync(Currency_Agent,"Convert 167 USD to Pkr")

print(result.final_output)
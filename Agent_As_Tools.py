from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel,trace,set_tracing_disabled
import asyncio
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

async def main():
    spanish_translater = Agent(
        name= "spanish_translater",
        instructions="Your task is to translate english to spanish",
        model = llm_model
    )
    
    french_translator = Agent(
        name= "French_translater",
        instructions="Your task is to translate english to french",
        model = llm_model
    )

    orchestrator = Agent(
        name = "Orchestrator",
        instructions="Your task is translation from english to spanish or french by using tools."
        "Call the tools as required."
        "If user doesnot specify  the language ask the user.",
        model= llm_model,
        tools = [spanish_translater.as_tool(tool_name="Spanish_translator", tool_description="You translate english to spanish"),
                french_translator.as_tool(tool_name="French_translator", tool_description="You translate english to french")]
    )

    result = await Runner.run(orchestrator,"Translate the following sentence to french: 'What a beautiful day!'")
    print(result.final_output)

asyncio.run(main())
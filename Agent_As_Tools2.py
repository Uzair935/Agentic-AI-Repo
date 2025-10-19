from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, trace, get_trace
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
    proofreader = Agent(
        name = "Proofreader",
        instructions = "Fix grammar and punctuation. Keep meaning. Reply only with the corrected text. ",
        model = llm_model
    )

    @function_tool
    async def proofreader_tool(given_text: str):
        """Fix grammar and punctuation. Keep meaning. Reply only with the corrected text."""
        result = await Runner.run(starting_agent=proofreader, input=given_text, max_turns=3)
        return(result.final_output)
    
    English_teacher = Agent(
        name = "English_Teacher",
        instructions="Help students write clearly. Use tools when asked to fix text.",
        model=llm_model,
        tools=[proofreader_tool]
    )

    result = await Runner.run(English_teacher, input="Please help me fix this sentence: She doesnot want any troble?")
    print(result.final_output)

asyncio.run(main())
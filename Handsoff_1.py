from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel,trace,set_tracing_disabled, handoff
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
    openai_client= external_client)

async def main():
    billing_agent = Agent(name= "Billing_agent",instructions="Help with billing", model= llm_model)
    refund_agent = Agent(name= "Refund_agent",instructions="Help with refund", model= llm_model)

    triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about billing, handoff to the Billing agent. "
        "If they ask about refunds, handoff to the Refund agent."
    ),
    handoffs=[billing_agent, refund_agent],
    model = llm_model
    )

    result = await Runner.run(triage_agent,input = "I need to check refund status")
    print(result.final_output)

asyncio.run(main())
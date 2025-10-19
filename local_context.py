import asyncio
from dataclasses import dataclass
from agents import Agent,Runner,RunContextWrapper,function_tool

@dataclass
class Userinfo:
    name:str
    id:int

@function_tool
async def fetch_age(wrapper: RunContextWrapper[Userinfo]) -> str:
    return f"User {wrapper.context.name} is 47 years old"

async def main():
    user_info = Userinfo(name="John",id=123)
    
    
    agent = Agent[Userinfo](
        name = "Assistant",
        tools = [fetch_age],
    )


    result = await Runner.run(
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel,trace,set_tracing_disabled, handoffs, RunContextWrapper
import asyncio
from dataclasses import dataclass,field
from copy import deepcopy
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
    @function_tool
    def calculator(expression:str):
        """Evaluate the expression"""
        try:
            return(eval(expression))
        except:
            return("Error")

    @dataclass
    class TaskState:
        """Shared memory between agents."""
        task:str
        plan:str = ""
        result:str = ""
        feedback:str = ""
        
    reporter = Agent(
        name="Reporter",
        instructions=(
            "The task is complete. Review the final 'result' and 'feedback' in the shared state. "
            "You are the final step, do not perform any further calculations or updates."),
            model= llm_model
    )
    critic = Agent(
        name="Critic",
        instructions=(
            "Review the executioner's result and write constructive feedback "
            "to improve clarity or correctness. Update the 'feedback' field in the shared state. "
            "Then hand off to the reporter to finalize the run."),
        model= llm_model,
        handoffs=[reporter],
        tools=[]
        )
    executioner =Agent(
        name = "Executioner",
        instructions=(
            "Read the 'plan' from the shared state and execute it using available tools. "
            "Store your final computed value in the 'result' field. "
            "Then hand off to the critic."
        ),
        tools= [calculator],
        model= llm_model,
        handoffs = [critic]
    )
    planner = Agent(
        name = "Planner",
        instructions=(
            "Read the user's task and create a detailed plan of steps to solve it. "
            "Store the plan in the 'plan' field, then hand off to the executioner agent."
        ),
        model= llm_model,
        handoffs=[executioner]
    )

    # @dataclass
    # class CustomRunContextWrapper(RunContextWrapper):
    #     state: TaskState = field(init=False)
    #     def __init__(self, state):
    #         super().__init__(state)
    #         self.state = state
    #     def copy(self):
    #         return CustomRunContextWrapper(deepcopy(self.state))
    #     def extend(self, value):
    #         """
    #         Provides the 'extend' method expected by the Runner utility, but does
    #         not call super() as the parent class doesn't implement it.
    #         """
    #         pass
    #     def __iter__(self):
    #         """
    #         Required by the Runner. Returns an iterator. 
    #         We yield an empty iterator as this wrapper only holds shared state.
    #         """
    #         return iter([])
        
    state= TaskState(task="What is (5 * 10) + (20 / 4)?")
    initial_task_prompt = f"The user's task is: {state.task}"
    # context = RunContextWrapper(state)
    result = await Runner.run(starting_agent=planner, input = initial_task_prompt)

    print("\n---Final Result---")
    print(f"Result: {result.final_output}")
    # print(f"Task: {state.task}")
    print(f"Plan: {state.plan}")
    # print(f"Critic Feedback: {state.feedback}")

asyncio.run(main())
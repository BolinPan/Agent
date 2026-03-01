import ast
from core.llm_client import LLMClient
from assets.prompt import PLANNER_PROMPT_TEMPLATE, EXECUTOR_PROMPT_TEMPLATE


# Planner implementation
class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    # Generate a step-by-step plan for solving the given question using the LLM
    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        # Call the LLM to generate the plan and extract it from the response, ensuring it's in the correct format (a Python list of strings)
        print("--- Generating Plan ---")
        response_text = self.llm_client.call(messages=messages) or ""
        print(f"Plan Generated:\n{response_text}")
        
        try:
            # Extract the Python list from the LLM's response and evaluate it to get the plan as a list of strings
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"Error parsing plan: {e}")
            print(f"Original response: {response_text}")
            return []
        except Exception as e:
            print(f"Unknown error occurred while parsing plan: {e}")
            return []


# Executor implementation
class Executor:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    # Execute the given plan step by step to solve the original question, using the LLM to get results for each step
    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        final_answer = ""
        
        print("\n--- Executing Plan ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> Executing step {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "None", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            
            # Call the LLM to execute the current step and get the result, then update history and final answer
            response_text = self.llm_client.call(messages=messages) or ""
            
            # Update history with the current step and its result, and set the final answer to the latest result
            history += f"Step {i}: {step}\nResult: {response_text}\n\n"
            final_answer = response_text
            print(f"Step {i} completed, result: {final_answer}")
        return final_answer


# Plan-and-Solve Agent implementation that integrates the Planner and Executor to solve complex problems in a structured way
class PlanAndSolveAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)
    
    # Main method to run the agent on a given question, which first generates a plan and then executes it to get the final answer
    def run(self, question: str):
        print(f"\n--- Starting to process question ---\nQuestion: {question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- Task Terminated ---\nUnable to generate a valid action plan.")
            return
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- Task Completed ---\nFinal answer: {final_answer}")


# Example usage
if __name__ == '__main__':
    try:
        # Initialize the LLM client and run the Plan-and-Solve agent on a sample question
        llm_client = LLMClient()
        agent = PlanAndSolveAgent(llm_client)

        # Run Plan-and-Solve agent on a sample question
        question = "A fruit shop sold 15 apples on Monday. On Tuesday, the number of apples sold was twice as much as Monday. On Wednesday, the number sold was 5 less than Tuesday. How many apples were sold in total over these three days?"
        agent.run(question)
    except Exception as e:
        print(f"Error running Plan-and-Solve agent: {e}")

from typing import List, Dict, Any
from urllib import response
from core.llm_client import LLMClient
from assets.prompt import INITIAL_PROMPT_TEMPLATE, REFLECT_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE


# A simple memory module to store the agent's execution attempts and reflection feedback, which can be used to build prompts for iterative improvement
class Memory:
    """
    A simple short-term memory module for storing the agent's actions and reflection trajectory.
    """
    def __init__(self):
        # Initialize an empty list to hold all records
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        Add a new record to memory.

        Args:
        - record_type (str): The type of record ('execution' or 'reflection').
        - content (str): The specific content of the record (e.g. generated code or reflection feedback).
        """
        # Append the new record to the records list and print a message indicating the memory has been updated
        self.records.append({"type": record_type, "content": content})
        print(f"📝 Memory updated, added a '{record_type}' record.")

    def get_trajectory(self) -> str:
        """
        Format all memory records into a coherent string for building prompts.
        """
        trajectory = ""
        # Iterate through all records and format them into a single string that can be used in prompts to provide context for the agent's iterative improvement process
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Previous attempt (code) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        Retrieve the most recent execution result (e.g., the latest generated code).
        """
        # Iterate through the records in reverse order to find the most recent 'execution' record and return its content, which can be used as context for reflection and refinement in the next iteration
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None


# Reflection agent implementation 
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations
    
    # Main method to run the agent on a given task, which includes an initial attempt followed by iterative reflection and refinement based on feedback until a satisfactory solution is generated or max iterations are reached
    def run(self, task: str):
        print(f"\n--- Starting task processing ---\nTask: {task}")

        # Initial attempt 
        print("\n--- Performing initial attempt ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # Iterative loop: reflection and refinement 
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

            # Reflection
            print("\n-> Reflecting...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # Check if we need to stop
            if "no need for improvement" in feedback.lower():
                print("\n✅ Reflection determined the code needs no improvement; task completed.")
                break

            # Refinement
            print("\n-> Refining...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        # After iterations, retrieve the final generated code from memory and print it as the final output of the task processing
        final_code = self.memory.get_last_execution()
        print(f"\n--- Task complete ---\nFinal generated code:\n```python\n{final_code}\n```")
        return final_code
    
    # A helper method used to call the LLM and obtain the full streaming response
    def _get_llm_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        # Ensure we handle cases where the generator might return None
        response_text = self.llm_client.call(messages=messages) or ""
        return response_text


# Example usage
if __name__ == '__main__':
    try:
        # Initialize the LLM client and run the Reflection agent
        llm_client = LLMClient()
        agent = ReflectionAgent(llm_client, max_iterations=2)

        # Run Reflection agent on a sample question
        task = "Write a Python function that finds all prime numbers between 1 and n."
        agent.run(task)
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        exit()


import re
from core.llm_client import LLMClient
from tools.search_tool import SearchTool
from tools.tool_executor import ToolExecutor
from assets.prompt import REACT_PROMPT_TEMPLATE


# ReAct agent implementation
class ReActAgent:
    def __init__(self, llm_client: LLMClient, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
    
    # Main loop to run the agent on a given question
    def run(self, question: str):
        self.history = []
        current_step = 0
        
        # Iteratively interact with the LLM and tools until we get a final answer or reach max steps
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---")
            
            # Construct the prompt with available tools and interaction history
            tools_desc = self.tool_executor.getAvailableTools()

            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)
            
            # Send the prompt to the LLM and get the response
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.call(messages=messages)
            if not response_text:
                print("Error: LLM failed to return a valid response.")
                break

            # Parse the LLM's response to extract Thought and Action
            thought, action = self._parse_output(response_text)
            if thought: print(f"Thought: {thought}")
            if not action:
                print("Warning: Failed to parse a valid Action, terminating.")
                break
            
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"\n--- Final answer ---")
                print(f"{final_answer}")
                return final_answer
            
            # Parse the Action to get the tool name and input, then execute the tool and observe the result
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: Invalid Action format, please check.")
                continue

            # Execute the tool and get the observation
            print(f"Action: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"Error：Cannot find the tool: '{tool_name}'."
            
            # Log the observation and update history for the next iteration
            print(f"Observation: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        
        # If we reach the maximum number of steps without getting a final answer, we terminate the process
        print("Reached maximum steps, terminating.")
        return None
    
    # Helper methods to parse the LLM's output
    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action
    
    # Parse the Action to extract the tool name and input
    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        return (match.group(1), match.group(2)) if match else (None, None)
    
    # Parse the Action when it is a Finish action to extract the final answer
    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""


# Example usage
if __name__ == '__main__':
    try:
        # Initialize the LLM client and tool executor, register tools
        llm_client = LLMClient()
        tool_executor = ToolExecutor()
        search_desc = "A web search engine. Use this tool when you need to answer questions about current events, facts, or information not found in your knowledge base."
        tool_executor.registerTool("SearchTool", search_desc, SearchTool)

        # Run ReAct agent on a sample question
        agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor)
        question = "What is Apple's latest phone model? What are its main selling points?"
        agent.run(question)
    except Exception as e:
        print(f"Error running ReAct agent: {e}")

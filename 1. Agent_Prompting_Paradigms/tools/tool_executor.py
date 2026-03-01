import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from search_tool import SearchTool

# Load environment variables from .env file
load_dotenv()

# Add the current directory to sys.path to ensure we can import local modules
sys.path.append(os.path.dirname(__file__))


# Construct a tool execution framework that can manage and execute various tools
class ToolExecutor:
    """
    A tool executor responsible for managing and executing tools.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    # Method to register a new tool in the toolbox
    def registerTool(self, name: str, description: str, func: callable):
        """
        Register a new tool in the toolbox.
        """
        if name in self.tools:
            print(f"Warning: Tool '{name}' already exists and will be overwritten.")
        
        # Register the tool with its name, description, and execution function
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool '{name}' has been registered.")

    # Method to get the execution function of a tool by name
    def getTool(self, name: str) -> callable:
        """
        Get the execution function of a tool by name.
        """
        return self.tools.get(name, {}).get("func")
    
    # Method to get a formatted description string of all available tools
    def getAvailableTools(self) -> str:
        """
        Get a formatted description string of all available tools.
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


# Example usage
if __name__ == '__main__':
    # Initialize the tool executor
    toolExecutor = ToolExecutor()

    # Register our practical search tool
    search_description = "A web search engine. Use this tool when you need to answer questions about current events, facts, or information not found in your knowledge base."
    toolExecutor.registerTool("SearchTool", search_description, SearchTool)
    
    # Print available tools
    print("\n--- Available Tools ---")
    print(toolExecutor.getAvailableTools())

    # Agent's Action call, this time we ask a real-time question
    print("\n--- Execute Action: SearchTool['What is Nvidia\'s latest GPU model'] ---")
    tool_name = "SearchTool"
    tool_input = "What is Nvidia's latest GPU model"

    # Get the tool function and execute it
    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- Observation ---")
        print(observation)
    else:
        print(f"Error: Tool '{tool_name}' not found.")

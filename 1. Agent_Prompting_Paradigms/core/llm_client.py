import os 
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict


# Load environment variables from .env file
load_dotenv()

# Define LLMClient class to interact
class LLMClient:
    """
    Construct LLMClient as a wrapper around the OpenAI API to facilitate interactions with language models. 
    It provides a method to call the LLM with a list of messages and handles streaming responses.
    """
    def __init__(self):
        self.model = os.getenv("GPT_MODEL_ID")
        self.api_key = os.getenv("OPENAI_API_KEY") 
        self.client = OpenAI(api_key=self.api_key)

    # Call method to send messages to the LLM and receive a response
    def call(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        try:
            # Create a chat for LLM iteration with streaming response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
             
            # Streaming response handling
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                # print(content, end="", flush=True)
                collected_content.append(content)
            return "".join(collected_content)
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return None


# Example usage
if __name__ == '__main__':
    try:
        # Initialize the LLM client
        LLMClient = LLMClient()
        
        # Example messages
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a Python function that takes a list of numbers and returns their sum."}
        ]
        
        # Call the LLM with the example messages
        responseText = LLMClient.call(exampleMessages)
    except ValueError as e:
        print(e)

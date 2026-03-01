import os
import sys
from dotenv import load_dotenv
from serpapi import SerpApiClient

# Load environment variables from .env file
load_dotenv()

# Add the current directory to sys.path to ensure we can import local modules
sys.path.append(os.path.dirname(__file__))


# Define the search function that uses SerpAPI to perform web searches
def SearchTool(query: str) -> str:
    """
    A practical web search engine tool based on SerpAPI.
    It intelligently parses search results and prioritizes returning direct answers or knowledge graph information.
    """
    print(f"Executing [SerpApi] web search: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY is not configured in the .env file."

        # Define search parameters
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "us",  # country code
            "hl": "en", # language code
        }

        # Create a SerpAPI client and get search results
        client = SerpApiClient(params)
        results = client.get_dict()

        # Intelligent parsing: prioritize finding the most direct answer
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # If there is no direct answer, return the summary of the first three organic results
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"Sorry, no information found about '{query}'."
    except Exception as e:
        return f"An error occurred during search: {e}"


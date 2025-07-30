import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_community.tools import BraveSearch

load_dotenv()
api_key = os.getenv("BRAVE_API_KEY")

search_tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 1})

mcp = FastMCP(
    name="web_search",
    version="1.0.0",
    description="web search capbility using Brave Search"
    
)

@mcp.tool()
async def search_web(query: str) -> str:
    """
    Search the web using Brave Search.
    """
    results = search_tool.invoke(query)
    if not results:
        return "No results found."
    
    # Format the results
    formatted_results = "\n".join([f"{i+1}. {result['title']}: {result['link']}" for i, result in enumerate(results)])
    return f"Search results:\n{formatted_results}"

if __name__ == "__main__":
    mcp.run()
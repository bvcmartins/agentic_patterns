# client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
import asyncio

model = ChatOllama(model="qwen3:32b")


server_params = StdioServerParameters(
    command="python",
    args=["server2.py"],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(f"Result: {result}")
            #tools = await load_mcp_tools(session)
            #agent = create_react_agent(model, tools)
            #agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return result

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)   
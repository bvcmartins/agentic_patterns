from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

server_params = StdioServerParameters(
    command="python",
    args=["server3.py"],
    env=None
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools.tools])
            result = await session.call_tool("add", {"a": 5, "b": 3})
            print("Result of add(5, 3):", result)

if __name__ == "__main__":
    asyncio.run(run())   
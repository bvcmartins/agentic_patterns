import asyncio
from langchain_ollama.chat_models import ChatOllama
from mcp_use import MCPAgent, MCPClient # Assuming mcp_use or similar library

async def main():
    # Configure MCP client to connect to your server
    config = {
        "mcpServers": {
            "my_math_server": {
                "command": "uvicorn", # Or direct path to server.py if run separately
                "args": ["server2:app", "--reload", "--host", "127.0.0.1", "--port", "3131"]
            }
        }
    }
    client = MCPClient.from_dict(config)



    #Initialize Ollama LLM
    llm = ChatOllama(model="qwen3:32b", base_url="http://127.0.0.1:11434")

    #Create an agent to manage interactions
    agent = MCPAgent(llm=llm, client=client)

    #Example query
    result = await agent.run("What is 15 multiplied by 7?")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
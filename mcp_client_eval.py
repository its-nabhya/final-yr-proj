import asyncio
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def get_mcp_answer(question: str) -> dict:
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Set GROQ_API_KEY in your .env file!")

    llm = ChatGroq(model="qwen/qwen3-32b")
    client = MCPClient.from_config_file("MCP-server/server.json")
    agent = MCPAgent(llm=llm, client=client, max_steps=5, memory_enabled=True)

    start_time = time.time()
    try:
        answer = await agent.run(question)
        # Extract all tool results as context
        context = ""
        if hasattr(agent, "tool_results") and agent.tool_results:
            context = "\n---\n".join([str(result) for result in agent.tool_results])
        else:
            context = "N/A"
        end_time = time.time()
        return {
            "answer": answer,
            "context": context,
            "response_time": end_time - start_time
        }
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is machine learning?"
    result = asyncio.run(get_mcp_answer(q))
    print("ANSWER:", result["answer"])
    print("CONTEXT:", result["context"])
    print("RESPONSE TIME:", result["response_time"])


import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat(config_file="MCP-server/server.json"):
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Set GROQ_API_KEY in your .env file!")

    # Create Groq LLM (choose appropriate model variant if needed)
    llm = ChatGroq(model="qwen/qwen3-32b")

    # Initialize MCP client
    client = MCPClient.from_config_file(config_file)

    # Create agent with tool/memory support
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=5,
        memory_enabled=True
    )

    print("\n===== Interactive MCP Chat (Groq only) =====\nType 'exit' or 'quit' to end.\nType 'clear' to clear history.\n=========================")
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation.")
                break
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared!")
                continue
            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
                #  Use .run() for streaming output
                # response = ""
                # async for chunk in agent.run(user_input):
                #     print(chunk, end="", flush=True)
                #     response += chunk
                # print() # Newline after response

            except Exception as e:
                print(f"Exception/Error: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    try:
        asyncio.run(run_memory_chat())
    except FileNotFoundError:
        print(f"\n[ERROR] Config file not found: {run_memory_chat.__defaults__[0]}")
        print("Are you sure the MCP server is running in the next terminal?")
    except KeyboardInterrupt:
        print("\nChat interrupted. Exiting.")
import asyncio
from mcp_chatbot import MCP_ChatBot

async def main_async():
    print("Starting MCP Chatbot...")
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main_async())

# Confirm Ollama is working via MAF

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool, HostedMCPTool
from agent_framework.openai import OpenAIChatClient
# from agent_framework.mcp.client import MCPClient

# Look here: https://github.com/microsoft/agent-framework/tree/main/python/packages/core

async def run_agent():
    agent = ChatAgent(
        chat_client = OpenAIChatClient(
            model_id="llama3.1:8b",
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        ),

        Instructions="You are a helpful assistant that specialises in summarising markdown files."
    )

    result = await agent.run("Tell me about markdown.")
    print(result)

if __name__ == "__main__":
    asyncio.run(run_agent())


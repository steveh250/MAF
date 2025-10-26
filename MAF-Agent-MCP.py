# Author: Steve Harris
# Purpose: Basic MAF agent using Ollama modles and docling via MCP
# NOTES:
#  - Refer to examples here: https://github.com/microsoft/agent-framework/tree/main/python/samples/getting_started/agents/openai

import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool, HostedMCPTool
from agent_framework.openai import OpenAIChatClient, OpenAIAssistantsClient, OpenAIResponsesClient

async def run_agent():
    # Initialize the OpenAI chat client (using Ollama endpoint)
    chat_client = OpenAIChatClient(
        model_id="llama3.1:8b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    # Create the MCP tool for Docling PDF parsing
    docling_tool= MCPStreamableHTTPTool(
        name="docling_tool",
        description="Convert a PDF to Markdown using docling MCP server",
        url="http://localhost:8000/mcp",  # MCP server URL
    )

    # Specify the paths to the md and pdf file
    pdf_file = "/home/ubuntu/MAF/MAF-Research.pdf"
    print(f"File Name: {pdf_file}\n\n")
    md_file = "/tmp/MAF-Research.md"
    print(f"File Name: {md_file}\n\n")

    # Create the agent with the chat client
    agent = chat_client.create_agent(
        name="PDF Parsing Agent",
        instructions="You are a helpful assistant that specializes in converting PDFs.",
        tools=[docling_tool]
    )

    # Run the agent with the MCP tool
    result = await agent.run(
        f"Using your tools convert the PDF document at {pdf_file} to MarkDown and save the markdown output to {md_file}.",
    )

    print(f"Agent: {result}\n\n")

if __name__ == "__main__":
    asyncio.run(run_agent())

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
        model_id="qwen3:8b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    # Create the MCP tool for Docling PDF parsing
    docling_tool= MCPStreamableHTTPTool(
        name="docling_tool",
        description="Convert a PDF to Markdown using docling MCP server",
        url="http://localhost:8000/mcp",  # MCP server URL
    )

    # Specify the paths to the pdf file
    pdf_file = "/home/ubuntu/MAF/MAF-Research.pdf"
    print(f"File Name: {pdf_file}\n\n")
    
    # Create the agent with the chat client
    agent = chat_client.create_agent(
        name="PDF Parsing Agent",
        instructions="You are a helpful assistant that specializes in converting PDFs. You have access to a docling_tool that uses docling to convert documents, you will ask docling to convert the document to a docling format, retrieve the key that identifies the converted document and then ask the docling_tool to convert the file to markdown and then display the markdown for the user.",
        tools=[docling_tool]
    )

    # Run the agent with the MCP tool
    result = await agent.run(
        f"Using your tools convert the PDF document at {pdf_file} to MarkDown.",
    )

    print(f"Agent: {result}\n\n")

if __name__ == "__main__":
    asyncio.run(run_agent())


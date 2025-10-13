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
    docling_tool = HostedMCPTool(
        name="Tool to convert files using docling via MCP",
        url="http://localhost:8000/mcp",
        approval_mode="never_require",
    )

    # Specify the path to the pdf file
    pdf_path = "/tmp/MCPTest.pdf"
    print(f"File Name: {pdf_path}\n\n")

    # Create the agent with the chat client
    agent = chat_client.create_agent(
        name="PDF Parsing Agent",
        instructions="You are a helpful assistant that specializes in parsing PDFs. You have access to a docling_parser tool that can convert PDFs to markdown. You must use the docling_tool to convert the files when prompted by the user.",
        tools=[docling_tool]
    )

    # Run the agent with the MCP tool
    result = await agent.run(
        f"Using your tools convert the PDF document at {pdf_path} to MarkDown and summarise the document",
        tools=[docling_tool]
    )

    print(f"Agent: {result}\n\n")


    # Start iterative conversation
    print("PDF Parsing Agent - Interactive Mode")
    print("Type 'exit' or 'quit' to end the conversation\n")

    # Interactive conversation loop
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Agent: Goodbye!")
            break

        if not user_input:
            continue

        # Continue the conversation with context
        result = await agent.run(user_input, tools=[docling_tool])
        print(f"Agent: {result}\n")

if __name__ == "__main__":
    asyncio.run(run_agent())

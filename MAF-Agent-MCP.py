import asyncio
import base64
from agent_framework import ChatAgent, MCPStreamableHTTPTool, HostedMCPTool
from agent_framework.openai import OpenAIChatClient, OpenAIAssistantsClient

async def run_agent():
    # Initialize the OpenAI chat client (using Ollama endpoint)
    chat_client = OpenAIChatClient(
        model_id="mistral-small",
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
    )

    # Create the MCP tool for Docling PDF parsing
    docling_tool = HostedMCPTool(
        name="docling_parser",
        url="http://localhost:8000/mcp",
    )

    # Specify the path to the pdf file
    pdf_path = "/tmp/MCPTest.pdf"
    print(f"File Name: {pdf_path}\n\n")

    # Create the agent with the chat client
    agent = chat_client.create_agent(
        name="PDF Parsing Agent",
        instructions="You are a helpful assistant that specializes in parsing PDFs. You have access to a docling_parser tool that can convert PDFs to markdown. Use this tool when asked to parse or analyze PDF files.",
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

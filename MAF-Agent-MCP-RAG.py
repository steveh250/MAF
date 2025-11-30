# Author: Steve Harris
# Purpose: MAF Agent with RAG that checks Company Info against an RFP
# NOTES:
#  - Requires: pip install chromadb

import asyncio
import uuid
import chromadb
from chromadb.utils import embedding_functions
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

# --- RAG Manager Class Implementation ---

class RAGManager:
    def __init__(self):
        self.client = chromadb.Client()
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="rfp_knowledge",
            embedding_function=self.ef
        )
    def reset_database(self):
        """Clears existing data to ensure fresh context."""
        try:
            self.client.delete_collection("rfp_knowledge")
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name="rfp_knowledge",
            embedding_function=self.ef
        )
        return "Database reset successfully."

    def add_document(self, content: str, source: str) -> str:
        """
        Adds text content to the vector database.
        Args:
            content: The text (markdown) to add.
            source: A label for where this text came from (e.g., "Company Profile", "RFP").
        """

        name="add_to_knowledge_base",
        description="Saves text to the vector DB. Requires 'content' and 'source' (e.g., 'Company Info' or 'RFP').",

        # Split by double newline to create logical chunks

        chunks = [c for c in content.split("\n\n") if c.strip()]

        if not chunks:
            return "No content found to add."

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source} for _ in chunks]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"--- [RAG] Ingested {len(chunks)} chunks from source: {source} ---")
        return f"Successfully added {len(chunks)} chunks from '{source}' to the knowledge base."

    def query_knowledge(self, query: str, n_results: int = 5) -> str:
        """
        Queries the vector database for relevant context.
        """

        name="query_knowledge_base",
        description="Searches the vector DB for answers. Useful for matching RFP requirements to Company capabilities.",

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        context = ""
        if results['documents']:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                source_label = meta.get('source', 'Unknown')
                context += f"-- Result {i+1} (Source: {source_label}) --\n{doc}\n\n"

        return context if context else "No relevant information found in the knowledge base."

async def run_agent():
    # 1. Initialize RAG and Reset DB
    rag_manager = RAGManager()
    rag_manager.reset_database()

    # 2. Initialize Chat Client
    chat_client = OpenAIChatClient(
        model_id="qwen3:8b-40k",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    # 3. Define Tools
    docling_tool = MCPStreamableHTTPTool(
        name="docling_tool",
        description="Convert a PDF to Markdown using docling MCP server",
        url="http://localhost:8000/mcp",
    )

    # 4. Define File Paths
    # The static company information (Capabilities, History, etc.)
    company_info_file = "/home/ubuntu/MAF/MyCompany-Capabilities-Test.pdf"
    # The specific RFP we are answering today
    rfp_file = "/home/ubuntu/MAF/MCPTest.pdf"

    print(f"Company Profile: {company_info_file}")
    print(f"RFP Target: {rfp_file}\n")

    # 5. Create Agent
    agent = chat_client.create_agent(
        name="RFP Response Architect",
        instructions="""You are an expert Bid Manager. Your goal is to write a winning response to an RFP.

        You have a specific workflow you must follow:
        1. INGESTION: You must read documents using 'docling_tool' and immediately save them to your database using 'add_to_knowledge_base'.
        2. RETRIEVAL: When answering questions, you must use 'query_knowledge_base' to find the specific RFP requirement AND the matching Company capability to ensure the answer is grounded in fact.
        """,
        tools=[docling_tool, rag_manager.add_document, rag_manager.query_knowledge]
    )

    # 6. Run the Agent with a Phased Prompt
    # We explicitly tell the agent to load the Company Info FIRST, then the RFP.
    result = await agent.run(
        f"""
        Step 1: Load the Company Information.
        Convert the file at '{company_info_file}' to markdown and add it to the knowledge base with source="Company Info".

        Step 2: Load the RFP.
        Convert the file at '{rfp_file}' to markdown and add it to the knowledge base with source="RFP".

        Step 3: Generate Responses.
        Scan the RFP data for the top 3 requirements. For each requirement:
        - Search the knowledge base for our company's matching capability.
        - Draft a response citing our specific experience.
        """
    )

    print(f"Agent: {result}\n\n")

if __name__ == "__main__":
    asyncio.run(run_agent())

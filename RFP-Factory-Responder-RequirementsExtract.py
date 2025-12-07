## Author: Steve Harris
# Purpose: MAF Agent with RAG that checks Company Info against an RFP
# NOTES:
#  - Requires: pip install chromadb python-docx

import asyncio
import uuid
import sys
import chromadb
from chromadb.utils import embedding_functions
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient
from docx import Document
from datetime import datetime
import logging
import os, re, json

# Suppress noisy MCP async generator cleanup warnings
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

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
        print(f"Database reset successfully.\n")
        return "Database reset successfully."

    def add_document(self, content: str, source: str) -> str:
        """
        Adds text content to the vector database.
        Args:
            content: The text (markdown) to add.
            source: A label for where this text came from (e.g., "Company Profile", "RFP").
        """

        name="add_to_knowledge_base",
        description="Saves text to the vector DB. Requires 'content' and 'source' (e.g., 'Company-Info' or 'RFP').",

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

    def query_knowledge(self, query: str, n_results: int = 10) -> str:
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

        print(f"Knowledge query executed: '{query}'\n")
        return context if context else "No relevant information found in the knowledge base."

def save_to_json(result, output_filename: str = "rfp_response.json"):
    """Extracts JSON payload for output file and saves model reasoning trace separately for audit."""

    # Extract text from Agent response
    if hasattr(result, 'text'):
        content = result.text
    elif hasattr(result, 'content'):
        content = result.content
    else:
        content = str(result)

    # Ensure audit filename ends in .audit.json
    base, ext = os.path.splitext(output_filename)
    audit_filename = f"{base}.audit.json"

    # --- Extract <think> reasoning content for audit log ---
    think_matches = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    reasoning_log = [t.strip() for t in think_matches if t.strip()]

    # --- Remove <think> blocks from main output content ---
    content_no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # --- Extract JSON fenced block if present ---
    json_str = None
    fence = "```json"
    if fence in content_no_think:
        start = content_no_think.index(fence) + len(fence)
        end = content_no_think.find("```", start)
        if end == -1:
            end = len(content_no_think)
        json_str = content_no_think[start:end].strip()
    else:
        # fallback: find braces
        first_brace = content_no_think.find("{")
        last_brace = content_no_think.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = content_no_think[first_brace:last_brace + 1].strip()

    # --- Parse JSON or fallback ---
    if json_str:
        try:
            data = json.loads(json_str)
        except Exception:
            data = {
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "raw_output": content_no_think
            }
    else:
        data = {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "raw_output": content_no_think
        }

    # --- Write main output JSON ---
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # --- Write reasoning audit log ---
    audit_record = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "reasoning_trace": reasoning_log,
        "raw_full_response": content  # full record for evidence
    }
    with open(audit_filename, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Extracted JSON saved to: {output_filename}")
    print(f"✓ Full audit + reasoning saved to: {audit_filename}")

    return output_filename, audit_filename

async def run_agent(rfp_file: str, output_file: str = "rfp_response.json"):
    # 1. Initialize RAG and Reset DB
    rag_manager = RAGManager()
    rag_manager.reset_database()

    # 2. Initialize Chat Client
    chat_client = OpenAIChatClient(
      # model_id="qwen3:8b-40k",
        model_id="qwen3:14b-40k",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    # 3. Define Tools
    docling_tool = MCPStreamableHTTPTool(
        name="docling_tool",
        description="Convert a PDF to Markdown using docling MCP server",
        url="http://localhost:8000/mcp",
    )

    print(f"RFP Target: {rfp_file}")
    print(f"Output File: {output_file}\n")

    # 4. Create Agent
    agent = chat_client.create_agent(
        name="RFP Response Architect",
        instructions=f"""
        You are an expert Bid Manager. Your job is to extract all requirements, scope of work, response information and questions from an RFP..

        TOOLS
        -----
        You have access to three important tools to convert documents (docling_tool) and manage the knowledge base (rag_manager.add_document and rag_manager.query_knowledge):

        1. docling_tool
           - Provides access to a Docling MCP server that allows you to convert documents, e.g. PDF files to markdown.
           - Always use this to read the source PDFs to add to the RAG knowledge base. Do not guess their contents.

        2. rag_manager.add_document
           - Saves markdown text into the knowledge base.
           - Parameters:
               - content: the markdown text you want to store.
               - source: "RFP".
           - Always call this after using docling_tool to ingest the document.

        3. rag_manager.query_knowledge
           - Retrieves relevant text chunks from the knowledge base using semantic search.
           - Parameters:
               - query: Wwhat you are looking for.
               - source_filter: Use "RFP" when searching for the exact RFP requirement text.
           - You must use this tool whenever you need factual details from the RFP. Do not rely on your memory.

        REQUIREMENT EXTRACTION
        ----------------------
        You must:
        - Work through the RFP and extract the Scope of work, proposal response format and requirements, evaluation criteria and anything else that looks like we need to respond to.
        - Show the requirements from the RFP as they appea rin the RFP, do not intepret them, show them exactly as they appear in the RFP.

        OUTPUT FORMAT
        -------------
        Output the requirements in a JSON format, retaining the RFP document hierarchy exactly as it appears in the RFP - section by section.

        GUARDRAILS
        ----------
        You must use the following guardrails when developing a respone:
        - Never fabricate anything that is not supported by retrieved information from the knowledge base.
        - If information is incomplete, say so clearly and suggest that the bid team add more detail.
        - Use professional, confident, and concise language suitable for a formal RFP.

        Your goal is to extract the requirements from the RFP that is fully traceable back to the source documents.
        """,
        tools=[docling_tool, rag_manager.add_document, rag_manager.query_knowledge]
    )

    # 5. Run the Agent with a Phased Prompt
    # We explicitly tell the agent to load the Company Info FIRST, then the RFP.
    try:
        result = await agent.run(
        f"""
        Use the workflow and tool usage described in your system instructions.

        INPUT FILES
        -----------
        - RFP-PDF: '{rfp_file}'

        TASK
        ----
        1. Perform your full ingestion phase for the RFP PDF:
        2. List all the of information you have extracted from the RFP PDF.

        The final output of this will be in a format that will be saved in a JSON document.
        """
        )


        print(f"Agent: {result}\n\n")

        return result
    finally:
        # Ensure proper cleanup of agent resources
        if hasattr(agent, 'cleanup'):
            await agent.cleanup()
        # Give async generators time to clean up
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Default file paths
    default_rfp = "/home/ubuntu/MAF/Sample-RFP-Managed-Services.pdf"
    default_output = "/home/ubuntu/MAF/rfp_requirements.json"

    # Parse command line arguments
    if len(sys.argv) >= 2:
        rfp_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else default_output
    else:
        print("Usage: python rfp_agent.py <rfp_pdf> [output_json]")
        print(f"\nUsing default files:")
        print(f"  RFP: {default_rfp}")
        print(f"  Output: {default_output}\n")
        rfp_file = default_rfp
        output_file = default_output

    # Run agent and save result
    result = asyncio.run(run_agent(rfp_file, output_file))

    # Save to JSON after async execution completes
    save_to_json(result, output_file)


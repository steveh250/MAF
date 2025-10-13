# Script to test the Docling MCP server with a PDF on the local filesystem
# Thanks to Claude Sonnet 4.5

import requests
import json

def parse_sse_response(response):
    """Parse Server-Sent Events response"""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    yield data
                except json.JSONDecodeError:
                    print(f"Could not parse: {line}")

headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json"
}

print("=== Testing Docling MCP Server - Complete Workflow ===\n")

# Step 1: Initialize
print("1. Initializing...")
response = requests.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    },
    stream=True
)

session_id = response.headers.get('mcp-session-id')
print(f"Session ID: {session_id}\n")

for data in parse_sse_response(response):
    break

headers['mcp-session-id'] = session_id

# Step 2: Send initialized notification
print("2. Sending initialized notification...")
response = requests.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    },
    stream=True
)
print(f"Status: {response.status_code}\n")

# Step 3: Convert the PDF document
print("3. Converting /tmp/MCPTest.pdf to Docling format...")
response = requests.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "convert_document_into_docling_document",
            "arguments": {
                "source": "/tmp/MCPTest.pdf"
            }
        }
    },
    stream=True
)

document_key = None
for data in parse_sse_response(response):
    print(f"Conversion result: {json.dumps(data, indent=2)}\n")

    # Extract document_key from result
    if 'result' in data and 'content' in data['result']:
        for content in data['result']['content']:
            if content.get('type') == 'text':
                text = content.get('text', '')
                # Try to parse as JSON to get document_key
                try:
                    result_data = json.loads(text)
                    document_key = result_data.get('document_key')
                    print(f"Document key: {document_key}\n")
                except:
                    pass

if not document_key:
    print("ERROR: Could not get document_key from conversion")
    exit(1)

# Step 4: Export to markdown
print(f"4. Exporting document {document_key} to markdown...")
response = requests.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "export_docling_document_to_markdown",
            "arguments": {
                "document_key": document_key
            }
        }
    },
    stream=True
)

for data in parse_sse_response(response):
    print(f"Markdown export result: {json.dumps(data, indent=2)}\n")

    # Extract and display markdown
    if 'result' in data and 'content' in data['result']:
        for content in data['result']['content']:
            if content.get('type') == 'text':
                text = content.get('text', '')
                try:
                    result_data = json.loads(text)
                    markdown = result_data.get('markdown', '')
                    print("=" * 60)
                    print("MARKDOWN OUTPUT:")
                    print("=" * 60)
                    print(markdown[:1000])  # Print first 1000 chars
                    if len(markdown) > 1000:
                        print(f"\n... (truncated, total length: {len(markdown)} chars)")
                except:
                    print(f"Markdown text: {text[:500]}")

print("\n✅ Complete!")

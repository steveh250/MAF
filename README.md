# RFP Factory

## Background
This started off as a dumping ground for experiments using the new Micrososft Agent Framework (MAF) to see if MAF would work offline.  The initial PoC involved Ollama (with Qwen3:8b), the MAF code and a Docling MCP server to parse out a PDF (an RFP) and generate answers.  I then added basic RAG (using Chromadb) to store the extracted text.  This developed into the direction I am now taking of building PoC's for both sides of an RFP factory (having been on both sides of the RFP process many, many times I know how painful both sides of the process can be).

The factory analogy has been used as I imagine these working in the background, e.g. via CRON jobs, monitoring folders for documents and automatically creating the responses or assessments and dropping the drafts into a folder for further human processing.

### Factory - RFP Response Generator
The thinking behind this is that it will truly be a factory, a self contained Python script that is woken up by a cron job (with a shell script wrapper), that looks in two folders for an RFP to process and company information, it then builds an RFP response into a Word document which it saves - useful for sales or pursuit teams to reduce the amount of time spent on preparing RFP responses.

### Factory - RFP Response Assessor
This is for the procurement teams that receive the RFP's and generate an initial assessment based on a rubric.  This is useful in improving the RFP assessment process to reduce effort (what typically happens is that procurement teams have an initial set of criteria to filter out responses before they are passed to the actual assessors but usually have limited expertise to do anything but a basic assessment) - this assement agent will help filter out weak responses as well as provide an initial assessment to the downstream human assessors.

# Development Tasks

## RFP Response Generator

## Phase 1 - MVP
 - [ ] Get rid of the async error messages.
 - [X] Add RAG (FAISS/Chroma) for input data to ground questions. (See MAF-Agent-MCP-RAG.py)
 - [ ] Add Word document creation to store output.
 - [ ] Pass files in as parameters.

### Phase 2 - Enhancements
 - Try larger Qwen models (qwen3:14b is also 4bit quantized,Q4_K_M, it should be able to be run in about 8GB with about 22GB needed for any decent context window size, qwen3:30b may be a bit too large with any large enough context window).
 - Add conversational, multi-turn (this doesn't quite fit the factory model but would be fun - would be more suited to an interactive solution).
 - Automate: Develop cron shell script to monitor folder for PDF's, process the PDF and generate response.
 - Add email support to send out the responses by email.

## RFP Response Assessor

### Phase 1 - MVP
 - [ ] Create basic assessor using the Response generator as a template - change the prompts.

# Obervations

## Ollama truncating prompts
Noticed this in the Ollama logs: 'Nov 02 22:57:57 ollama[1011]: time=2025-11-02T22:57:57.174Z level=WARN source=runner.go:159 msg="truncating input prompt" limit=4096 prompt=7268 keep=4 new=4096'
 - Didn't seem able to use things like extra_body in the Python code for the Ollama setup to be able to modify the context window size.
 - Found this article: https://github.com/ollama/ollama/issues/8099
 - Applied these changes - including changing the model name in the Python script and worked much better.

```
   		(venv) ubuntu:~/MAF$ ollama run qwen3:8b
		>>> /set parameter num_ctx 40960
		Set parameter 'num_ctx' to '40960'
		>>> /save qwen3:8b-40k
		Created new model 'qwen3:8b-40k'
		>>> /bye
		(venv) ubuntu:~/MAF$ ollama list
		NAME                 ID              SIZE      MODIFIED
		qwen3:8b-40k         b891e3e3f240    5.2 GB    7 seconds ago
		qwen3:8b             500a1f067a9f    5.2 GB    59 minutes ago
		gemma3:270m          e7d36fb2c3b3    291 MB    4 weeks ago
		gemma3-doc:latest    f3ad5bc8c220    291 MB    2 months ago
```

```
	(venv) ubuntu@:~/MAF$ ollama show qwen3:8b-40k
  	Model
    	architecture        qwen3
    	parameters          8.2B
    	context length      40960
    	embedding length    4096
    	quantization        Q4_K_M

  	Capabilities
    	completion
    	tools
    	thinking

  	Parameters
    	num_ctx           40960
```
**NOTE: num_ctx is not present in the base model for some reason.**






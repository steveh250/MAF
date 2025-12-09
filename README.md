# RFP Factory

## Background
This started off as a dumping ground for experiments using the new Micrososft Agent Framework (MAF) to see if MAF would work offline.  The initial PoC involved Ollama (with Qwen3:8b), the MAF code and a Docling MCP server to parse out a PDF (an RFP) and generate answers.  I then added basic RAG (using Chromadb) to store the extracted text.  This developed into the direction I am now taking of building PoC's for both sides of an RFP factory (having been on both sides of the RFP process many, many times I know how painful both sides of the process can be).

The factory analogy has been used as I imagine these working in the background, e.g. via CRON jobs, monitoring folders for documents and automatically creating the responses or assessments and dropping the drafts into a folder for further human processing.

### Factory - RFP Response Generator
The thinking behind this is that it will truly be a factory, a self contained Python script that is woken up by a cron job (with a shell script wrapper), that looks in two folders for an RFP to process and company information, it then builds an RFP response into a Word document which it saves - useful for sales or pursuit teams to reduce the amount of time spent on preparing RFP responses.

### Factory - RFP Response Assessor
This is for the procurement teams that receive the RFP's and generate an initial assessment based on a rubric.  This is useful in improving the RFP assessment process to reduce effort and increase throughput (what typically happens is that procurement teams have an initial set of criteria to filter out responses before they are passed to the actual assessors but usually have limited expertise to do anything but a basic assessment) - this assement agent will help filter out weak responses as well as provide an initial assessment to the downstream human assessors.

# Development Tasks

## RFP Response Generator

## Phase 1 - PoC MVP
 - [X] Get rid of the async error messages.
 - [X] Add RAG (Chroma) for input data to ground questions.
 - [X] Add Word document creation to store output.
 - [X] Pass files in as parameters (with defaults).
 - [X] Try larger Qwen models (14b runs just fine - needed context window modified).
 - [X] Improved prompts to reduce instructtion and user prompt overlap (got better responses and less reasoning).
 - [X] Built version to extract RFP contents into JSON and then loop through the individual JSON sections and develop responses (replicates what I do manually with GenAI's), all consolidated into a final word document.  This worked really well and was a major enhancement.
 	- [X] Keep JSON as intermediate format as it's entriely possible the RFP reponse could feed into an automated assessor that would ingest the JSON and assess it (easier to assess than parsing a word document).

### Phase 2 - Enhancements
 - Fix the company information ingestion and RAG extraction - the model is still hallucinating.
 - Try larger Qwen models - qwen3:30b may be a bit too large with any large enough context window, although I suspect we will get a better quality RFP respnse from this larger model).
 - Add conversational, multi-turn (this doesn't fit the factory model but would be fun - would be more suited to an interactive solution).
 - Automate: Develop cron shell script to monitor folder for PDF's, process the PDF and generate response (whether that is drafting a response or assessing a rubric).
 - Add email support to send out the responses by email.
 - I suspect the basic rag is insufficient to make this production ready - maybe even use A2A and MAO to have a seperate agent answer the questions (may be getting to tool overload).
 - QA
 	- Add a QA portion to the prompt - persona based.
  	- Add ability to pull in files that describe the customer and store them in ChromaDB to support the persona.

## RFP Response Assessor

### Phase 1 - PoC MVP
 - [X] Create basic assessor using the Response generator as a template - change the prompts.

# Observations

## Capturing Reasoning
Initially I didn't want to capture the reasoning information in the Word document but quickly changed my mind as:
 - _Prompt performance_: Wathcing the Ollama logs and seeing the number and duration of the inferences I could see, combined with the reasoning output, that there is an opportunity to improve the performance by making the prompts clearer.
 - _Transparency_: For the person receiving the output the reasoning provides some insight into the models reasoning process.

## Model Size
I optimized the prompt as much as possible and the 8B model was giving good answers.  I upgraded to the Qwen:14B model and it made a massive difference - thoght more clearly and when it encountered errors provided much clearer explanations of the errors and selected tools more clearly.

## Docling Failures
I was getting Docling failuers and it looked like the model was calling Docling with an incorrect max_size format.  Added the following to the agent prompt to solve that.
```
https://github.com/steveh250/MAF-RFP-Factory/blob/main/RFP-Factory-Responder.py
```
## RAG Problem when used for RFP Processing
For version 4b11400 I worked on forcing the agent to load the RFP into the vector database but it didn't do a very good job at all of extracting the requirements from the vector database - it did a much better job when it seemed to just process the Docling markdown.  Reverted back to a previous version and worked on ensuring the company information was stored in the vector database.

## Question Answering
It became apparent, even after upgrading to the 14B model that the approach of answering all the RFP requirements in one go by an agent wasn't going to work (maybe a function of the context window size of the smaller models).  Took a different approach of using the model to extract the requirements and then process them one at a time (an approach I have had a lot of success with when manually responding to RFP's and Grant Requests using models) - this improved the quality of the output dramatically and it also, as expected, took longer to generate a document as the model was being called for each JSON element (RFP requirement).

## Ollama truncating prompts
Noticed this in the Ollama logs: 'Nov 02 22:57:57 ollama[1011]: time=2025-11-02T22:57:57.174Z level=WARN source=runner.go:159 msg="truncating input prompt" limit=4096 prompt=7268 keep=4 new=4096'
 - Didn't seem able to use things like extra_body in the Python code for the Ollama setup to be able to modify the context window size.
 - Found this article: https://github.com/ollama/ollama/issues/8099
 - Applied these changes - including changing the model name in the Python script and worked much better.
 - Also saw the same message with the 14B model, used same approach to fix.

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






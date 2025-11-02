# DUMPING GROUND FOR EXPERIMENT NOTES

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
**NOTE: num_ctx is not present in the base model for some reason**






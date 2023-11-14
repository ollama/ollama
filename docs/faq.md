# FAQ

## How can I view the logs?

On macOS:

```
cat ~/.ollama/logs/server.log
```

On Linux:

```
journalctl -u ollama
```

If you're running `ollama serve` directly, the logs will be printed to the console.

## How can I expose Ollama on my network?

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

On macOS:

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

On Linux:

Create a `systemd` drop-in directory and set `Environment=OLLAMA_HOST`

```bash
mkdir -p /etc/systemd/system/ollama.service.d
echo "[Service]" >>/etc/systemd/system/ollama.service.d/environment.conf
```

```bash
echo "Environment=OLLAMA_HOST=0.0.0.0:11434" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## How can I allow additional web origins to access Ollama?

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable:

On macOS:

```bash
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

On Linux:

```bash
echo "Environment=OLLAMA_ORIGINS=http://129.168.1.1:*,https://example.com" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## Where are models stored?

- macOS: Raw model data is stored under `~/.ollama/models`.
- Linux: Raw model data is stored under `/usr/share/ollama/.ollama/models`



Below the models directory you will find a structure similar to the following:

```shell
.
├── blobs
└── manifests
   └── registry.ollama.ai
      ├── f0rodo
      ├── library
      ├── mattw
      └── saikatkumardey
```

There is a `manifests/registry.ollama.ai/namespace` path. In example above, the user has downloaded models from the official `library`, `f0rodo`, `mattw`, and `saikatkumardey` namespaces. Within each of those directories, you will find directories for each of the models downloaded. And in there you will find a file name representing each tag. Each tag file is the manifest for the model.  

The manifest lists all the layers used in this model. You will see a `media type` for each layer, along with a digest. That digest corresponds with a file in the `models/blobs directory`.

### How can I change where Ollama stores models?

To modify where models are stored, you can use the `OLLAMA_MODELS` environment variable. Note that on Linux this means defining `OLLAMA_MODELS` in a drop-in `/etc/systemd/system/ollama.service.d` service file, reloading systemd, and restarting the ollama service.

## Does Ollama send my prompts and answers back to Ollama.ai to use in any way?

No. Anything you do with Ollama, such as generate a response from the model, stays with you. We don't collect any data about how you use the model. You are always in control of your own data.

## Does Ollama use the OpenAI API?

As of November 14, 2023, Ollama does not use the OpenAI API. The Ollama API is documented at [jmorganca/ollama/docs/api](./api) and works differently from how the OpenAI API works. We are investigating other options that may offer more compatibility. Until then, some of our users have had success with using [LiteLLM](https://docs.litellm.ai/docs/providers/ollama) in front of Ollama. 
## How can I use Ollama in VSCode to help me code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. You can see the list of plugins at the bottom of the main repository readme.
## How do I send a document to Ollama?

Ollama is a tool that will send a prompt to a model and return the generated answer. On its own, it doesn't know about documents. But as a developer you can add that capability. Or you can use the work of other developers who have leveraged Ollama to achieve this. Review the list of integrations others have already built at the bottom of the main repository readme if you just want to use someone else's project to do this.  

If you are looking to build your own solution, start by looking at Retrieval Augmented Generation, or RAG. This breaks apart the document into smaller pieces, then generates embeddings for the pieces that are stored in a vector database. An embedding converts the words into a series of numbers. When you ask your application to answer a question, that question is also converted into an embedding and then compared to each of the chunks in the database. The most similar chunks can then be fed to the model as source material to answer the question.

The reason we do this is that most models have a limited context which is the input size, and your document is often bigger. Plus, often providing too much information will simply confuse the model and result in wrong answers.

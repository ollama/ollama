# Ollama Deep Dive

Ollama is a tool for running Large Language Models locally on whatever hardware you own. This guide assumes you have installed Ollama and are eager to start understanding how it works. 

Before getting started, ensure you also have the tool JQ installed. You can find more about it here: https://jqlang.github.io/jq/

The first step to working with Ollama from the API is to pull a model.

```shell
curl http://localhost:11434/api/pull -d '{
  "name": "llama2"
}' 
```

*(The CLI alternative is the command `ollama pull llama2`.)*

You should see something like this output from the API:

```javascript
{"status":"pulling manifest"} 
{"status":"pulling 22f7f8ef5f4c","digest":"sha256:22f7f8ef5f4c791c1b03d7eb414399294764d7cc82c7e94aa81a1feb80a983a2","total":3825807040,"completed":3825807040}
{"status":"pulling 8c17c2ebb0ea","digest":"sha256:8c17c2ebb0ea011be9981cc3922db8ca8fa61e828c5d3f44cb6ae342bf80460b","total":7020,"completed":7020}
{"status":"pulling 7c23fb36d801","digest":"sha256:7c23fb36d80141c4ab8cdbb61ee4790102ebd2bf7aeff414453177d4f2110e5d","total":4766,"completed":4766}
{"status":"pulling 2e0493f67d0c","digest":"sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988","total":59,"completed":59}
{"status":"pulling 2759286baa87","digest":"sha256:2759286baa875dc22de5394b4a925701b1896a7e3f8e53275c36f75a877a82c9","total":105,"completed":105}
{"status":"pulling 5407e3188df9","digest":"sha256:5407e3188df9a34504e2071e0743682d859b68b6128f5c90994d0eafae29f722","total":529,"completed":529}
{"status":"verifying sha256 digest"}
{"status":"writing manifest"}
{"status":"removing any unused layers"}
{"status":"success"}
```

Docker was able to work its magic with the concept of **layers**. And it's the same idea here. If you haven't pulled llama2 before, you may see it actually download the model.

The first time you **pull** a model, the entire model will be downloaded. If you repeat that, Ollama will ensure the files are different and only download files that have changed. The **digest** in the output above helps us understand that. And the digest happens to be the name of the layer or file that was downloaded.

Let's find these files that were downloaded. In a second terminal window, navigate to the Ollama data root directory. On Mac, this will be in `~/.ollama`. On Linux and WSL, you can find this at `/usr/share/ollama/.ollama`. From there go to `models/manifests/registry.ollama.ai/library`. What do you see in this directory? You can list the directory with this command:

```shell
ls
```

You should see a folder for every model you have pulled. Now go into `llama2`. In this directory, you can see a file for each variation of **llama2** pulled. Now we want to look at the file:

```shell
cat latest | jq
```

Here we can see each of the layers of this model. And you can see that the big file, as indicated by the size, is the model. There are also a couple of licenses, a template layer, and a params layer. Notice that the digest for each of those appears in the other terminal that showed the output from our pull operation.

Now try pulling a different model:

```shell
curl http://localhost:11434/api/pull -d '{
  "name": "mattw/nothingnew"
}'
```

In a third terminal window, lets go to `.ollama/models/manifests/registry.ollama.ai/mattw/nothingnew`. Look at the latest file:

```shell
cat latest | jq
```

This is a different model, but this file looks a little different but some numbers are the same. I copied the model and added a seed parameter. So the model and template digests are the same as **llama2**, but the params digest is different.

Now let's take a look at the modelfile:

```shell
curl http://localhost:11434/api/show -d '{
    "name": "mattw/nothingnew"
  }' | jq
```

*(The CLI alternative of this command is `ollama show --modelfile mattw/nothingnew`)*

Leave this terminal and go to any of the other terminals you have open, navigate to `.ollama/models/blobs`. You will see a list of files labelled `sha256:` and then a string of characters. When we showed the manifest, the template layer had a manifest of `sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988` so in the **blobs** directory run this:

```shell
cat sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988
```

Notice that this is the same template we saw when we showed the modelfile.

There are a few things we have learned so far. Copying a model or creating a new model based on another model in the library only creates a new manifest and any unique layers. The actual model weights file is the same, and we share that file across all models that use it. All the layers are named based on the sha256 digest of the file. And most of the files (except the model and adapters) are text files. Another key concept you saw was that in Ollama, a model is not just the raw model weights binary file, but also the template, the system prompt, the parameters, and potentially more. For more information about creating models and modelfiles, see the [modelfile documentation](./modelfile.md).

<details>
  <summary><strong>Learn more about where Ollama stores files</strong></summary>

> Ollama creates a folder structure to store all of its data:
> - macOS: All data is stored under `~/.ollama`.
> - Linux: All data is stored under `/usr/share/ollama/.ollama`
>
> ```shell
> .
> ├── logs
> └── models
>     ├── blobs
>     └── manifests
>       └── registry.ollama.ai
>           ├── f0rodo
>           ├── library
>           ├── mattw
>           └── saikatkumardey
> ```
>
> At the root of the Ollama folders is the history file and the SSH keys. The history file lists all the prompts and commands you have used in the CLI. This file is never share with anyone and stays on your machine. The SSH keys are 
>
> There is a `manifests/registry.ollama.ai/namespace` path. In example above, the user has downloaded models from the official `library`, `f0rodo`, `mattw`, and `saikatkumardey` namespaces. Within each of those directories, you will find directories for each of the models downloaded. And in there you will find a file name representing each tag. Each tag file is the manifest for the model.  
>
> The manifest lists all the layers used in this model. You will see a `media type` for each layer, along with a digest. That digest corresponds with a file in the `models/blobs directory`.
>
> ## How can I change where Ollama stores models?
>
> To modify where models are stored, you can use the `OLLAMA_MODELS` environment variable. Note that on Linux this means defining `OLLAMA_MODELS` in a drop-in `/etc/systemd/system/ollama.service.d` service file, reloading systemd, and restarting the ollama service.
>
> On a Mac, you will need to stop the Ollama service by clicking the icon in the menubar and choosing **Quit Ollama**. Then in a new terminal, run `OLLAMA_MODELS=<my/new/folder> ollama serve`.

</details>
&nbsp;



## Ollama is 100% local

Apart from installing Ollama and pulling models from the library, nothing else in Ollama ever reaches out anywhere. Everything you ask a model stays local and will never leave your machine. None of your prompts or the generated output is ever used to improve the models either.

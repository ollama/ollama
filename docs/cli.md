# Ollama CLI

There are two main functions of the Ollama CLI. First it runs the service on the backend. This is the process that actually interacts with the models. Then it's the main UI for the user, passing commands to the service to run.

```bash
❯ ollama
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```

## Using Ollama interactively

The main command you use with Ollama is **`ollama run <modelfile>`**. This will put you into the Ollama REPL (Read - Eval - Print Loop). A REPL is a common environment with programming languages such as Python which lets you interactively run instructions and see the output.

If you know your input will cross multiple lines, you can wrap the text with three double-quotes (**`"""`**). But you can paste any multiline text from the clipboard without needing to wrap it first. 

### Slash Commands

When in the REPL you can use slash commands to show and set various settings. Type **`/?`** to see the top level of options, then you can try just **`/set`** or **`/show`** to see all of the lower level options. For instance, if you want to show the modelfile, run **`/show modelfile`**. Then you can run **`set system You are a helpful AI assistant who always answers in rhyiming couplets.`**. Now, depending on the capabilities of the model, when you ask a question you will get an answer in the form of a rhyming poem.

Another important slash command is **`/set verbose`** which is the equivalent of using the verbose flag (**`ollama run llama2 --verbose`**). This will show you the performance of the model for this specific generation.

Experiment with the other slash commands to see what is possible.

## Using Ollama non-interactively

The other way to generate output is to pass the prompt as a command line argument. 

```bash
ollama run llama2 "why is the sky blue"
```

This will run the generation and return to the command prompt when completed. Depending on the context size of the model, you may also be able to provide the text of a file here, using the features of your shell, such as **`ollama run llama2 "How can I make this code better: $(cat ./myfunction.py)`** or **`cat main.go | ollama run llama2 "how can i make this code better"`**, both of which do the same thing.

## Using Multimodal models

There are a few multimodal models, such as **Llava** and **Bakllava**. To use these, specify a prompt and in the case of llava models, a path to an image: **`>>> What's in this image? /Users/jmorgan/Desktop/smile.png`**

## Using JSON mode

Occasionally, when working with models, you need the output to be well-formed JSON. This is especially useful when the output will be used to call functions in your application. To accomplish this, use the **`--format json`** flag on the CLI or the **`/set format json`** slash command. Then make sure to also specify that you want to output JSON in the prompt.

To improve the quality of the output, consider specifying your desired schema in the prompt. Also, try using few-shot prompting to give the model examples of what the desired output looks like.

## Creating models

You can choose to modify any existing model by creating a new **Modelfile** and setting the instructions you want to change. For instance, to make the poetic llama2 we used above, you could create a Modelfile like this:

```plaintext
FROM llama2
SYSTEM You are a helpful AI assistant who always answers in rhyiming couplets.
```

Then create the model using **`ollama create llamaspeare -f ./Modelfile`**, where llamaspeare is the name I want to use for the model and `./Modelfile` is the path to the Modelfile above.

If you want to push the model to the Registry, you will need to add your namespace (visit [https://ollama.ai/signup](Signup) to create your namespace). You can do that with **`ollama cp llamaspeare yournamespace/llamaspeare`**.

Now you can push with **`ollama push yournamespace/llamaspeare`**.

## Managing models

To download any model from the Library, run **`ollama pull llama2`**. Or you can run **`ollama run llama2`** which will download the model and then drop you in the REPL.

If you have too many models downloaded, you can remove them with **`ollama rm llama2`**. Any file in the model is only deleted when there are no more references to that file.

### Where does Ollama store its files

Ollama creates a folder structure to store all of its data:

- macOS: All data is stored under `~/.ollama`.
- Linux: All data is stored under `/usr/share/ollama/.ollama`

```shell
.
├── logs
└── models
   ├── blobs
   └── manifests
     └── registry.ollama.ai
         ├── f0rodo
         ├── library
         ├── mattw
         └── saikatkumardey
```

At the root of the Ollama folders is the history file and the SSH keys. The history file lists all the prompts and commands you have used in the CLI. This file is never shared with anyone and stays on your machine. The SSH keys are used when you push models to your namespace on Ollama.ai.

There is a `manifests/registry.ollama.ai/namespace` path. In the example above, the user has downloaded models from the official `library`, `f0rodo`, `mattw`, and `saikatkumardey` namespaces. Within each of those directories, you will find directories for each of the models downloaded. And in there you will find a file name representing each tag. Each tag file is the manifest for the model.  

The manifest lists all the layers used in this model. You will see a `media type` for each layer, along with a digest. That digest corresponds with a file in the `models/blobs directory`.

## How can I change where Ollama stores models?

To modify where models are stored, you can use the `OLLAMA_MODELS` environment variable. Note that on Linux this means defining `OLLAMA_MODELS` in a drop-in `/etc/systemd/system/ollama.service.d` service file, reloading systemd, and restarting the ollama service.

On a Mac, you will need to stop the Ollama service by clicking the icon in the menubar and choosing **Quit Ollama**. Then in a new terminal, run `OLLAMA_MODELS=<my/new/folder> ollama serve`.

## Serving Ollama

By default, using the installation script on Linux, or using the built executable for Mac, `ollama serve` runs automatically. But if you built Ollama yourself or in a few other circumstances, there are reasons you may choose to modify how it runs. On Linux, you may opt to update the Ollama service files, and on all the platforms you may want to run it in a terminal interactively.

### How to use environment variables

Ollama uses environment variables to enable different functionalities. But you need to use them in the right way to get any value from them.

#### Using Ollama server environment variables on Mac

On macOS, Ollama runs in the background and is managed by the menubar app. If you want to add environment variables, you will need to manually run Ollama.

1. Click the menubar icon for Ollama and choose **Quit Ollama**.
2. Open a new terminal window and run the following command (this example uses `OLLAMA_HOST` with an IP address of `123.1.1.1`):

   ```bash
   OLLAMA_HOST=123.1.1.1 ollama serve
   ```

#### Using Ollama server environment variables on Linux

If you installed Ollama with the install script, a systemd service was created, running as the Ollama user. To add an environment variable, such as OLLAMA_HOST, you will need to follow these steps:

1. Create a `systemd` drop-in directory and add a config file. You only need to do this once.

   ```bash
   mkdir -p /etc/systemd/system/ollama.service.d
   echo '[Service]' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

2. For each environment variable you wish to use add it to the config file:

   ```bash
   echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

3. Reload `systemd` and restart Ollama:

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```

### Setting the models directory

When Ollama downloads models, it uses its home directory to store them. This is in a different location on macOS vs. Linux.

- macOS: All model data is stored under `~/.ollama/models`.
- Linux: All model data is stored under `/usr/share/ollama/.ollama/models`

If you want to use a different directory, set the environment variable `OLLAMA_MODELS` to the directory you wish to use. Refer to the section above for how to use environment variables on your platform.

### Exposing Ollama on a Network

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable. Refer to the section above for how to use environment variables on your platform.

### Allowing additional web origins to access Ollama

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable. For example to add all ports on 192.168.1.1 and https://example.com, you would use:

```shell
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com
```

Refer to the section above for how to use environment variables on your platform.

### Using Ollama behind a proxy

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. When using either variables, ensure it is set where `ollama serve` can access the values. When using `HTTPS_PROXY`, ensure the proxy certificate is installed as a system certificate. Refer to the section above for how to use environment variables on your platform.

#### Using Ollama behind a proxy in Docker

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container.

Alternatively, the Docker daemon can be configured to use a proxy. Instructions are available for Docker Desktop on [macOS](https://docs.docker.com/desktop/settings/mac/#proxies), [Windows](https://docs.docker.com/desktop/settings/windows/#proxies), and [Linux](https://docs.docker.com/desktop/settings/linux/#proxies), and Docker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy).

Ensure the certificate is installed as a system certificate when using HTTPS. This may require a new Docker image when using a self-signed certificate.

```dockerfile
FROM ollama/ollama
COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificates
```

Build and run this image:

```shell
docker build -t ollama-with-ca .
docker run -d -e HTTPS_PROXY=https://my.proxy.example.com -p 11434:11434 ollama-with-ca
```

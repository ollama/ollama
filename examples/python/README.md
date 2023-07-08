# Python

This is a simple example of calling the Ollama api from a python app.

First, download a model:

```
curl -L https://huggingface.co/TheBloke/orca_mini_3B-GGML/resolve/main/orca-mini-3b.ggmlv3.q4_1.bin -o orca.bin
```

Then run it using the example script. You'll need to have Ollama running on your machine.

```
python3 main.py orca.bin
```

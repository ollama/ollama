# PrivateGPT with Llama 2 uncensored

https://github.com/ollama/ollama/assets/3325447/20cf8ec6-ff25-42c6-bdd8-9be594e3ce1b

> Note: this example is a slightly modified version of PrivateGPT using models such as Llama 2 Uncensored. All credit for PrivateGPT goes to Iván Martínez who is the creator of it, and you can find his GitHub repo [here](https://github.com/imartinez/privateGPT).

### Setup

Set up a virtual environment (optional):

```
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```shell
pip install -r requirements.txt
```

Pull the model you'd like to use:

```
ollama pull llama2-uncensored
```

### Getting WeWork's latest quarterly earnings report (10-Q)

```
mkdir source_documents
curl https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf -o source_documents/wework.pdf
```

### Ingesting files

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

### Ask questions

```shell
python privateGPT.py

Enter a query: How many locations does WeWork have?

> Answer (took 17.7 s.):
As of June 2023, WeWork has 777 locations worldwide, including 610 Consolidated Locations (as defined in the section entitled Key Performance Indicators).
```

### Try a different model:

```
ollama pull llama2:13b
MODEL=llama2:13b python privateGPT.py
```

## Adding more files

Put any and all your files into the `source_documents` directory

The supported extensions are:

- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.enex`: EverNote,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.msg`: Outlook Message,
- `.odt`: Open Document Text,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.ppt` : PowerPoint Document,
- `.txt`: Text file (UTF-8),

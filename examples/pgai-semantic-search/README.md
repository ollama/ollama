# pgai + Ollama Example


[pgai](https://github.com/timescale/pgai) is an open-source postgres extension that simplifies the process of building search, retrieval augmented generation (RAG), and other AI applications.

This example demonstrates how to use [pgai vectorizer](https://www.timescale.com/blog/vector-databases-are-the-wrong-abstraction/) with ollama to automatically create embeddings for text and perform semantic search on them. with ollama to automatically create embeddings for text and perform semantic search on them.

## Prerequisites

1. Docker and Docker Compose installed
2. Python 3.8+

## Setup

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

2. Start the docker compose services:
   ```bash
   docker compose up -d
   ```
3. Run the example:
   ```bash
   python main.py
   ```

## What's happening?

The example code:
1. Starts Ollama and downloads an embedding model
2. Connects to PostgreSQL and:
   1. Installs the pgai extension
   2. Creates a sample blog table with data
   3. Sets up a "vectorizer" to automatically create embeddings for the blog posts
   4. Performs a semantic search query to find blog posts related to "good food"

If you don't want to use python, you can also follow the process step by step in pgais quickstart tutorial [here](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md).

# News Summarizer

This example goes through a series of steps:

  1. You choose a topic area (e.g., "news", "NVidia", "music", etc.).
  2. Gets the most recent articles on that topic from various sources.
  3. Uses Ollama to summarize each article.
  4. Creates chunks of sentences from each article.
  5. Uses Sentence Transformers to generate embeddings for each of those chunks.
  6. You enter a question regarding the summaries shown.
  7. Uses Sentence Transformers to generate an embedding for that question.
  8. Uses the embedded question to find the most similar chunks.
  9. Feeds all that to Ollama to generate a good answer to your question based on these news articles.

This example lets you pick from a few different topic areas, then summarize the most recent x articles for that topic. It then creates chunks of sentences from each article and then generates embeddings for each of those chunks.

You can run the example like this:

```bash
pip install -r requirements.txt
python summ.py
```

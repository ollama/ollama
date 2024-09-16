import json
import requests
from sentence_transformers import SentenceTransformer
from mattsollamatools import chunker
import requests
import json
from newspaper import Article
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mattsollamatools import chunker
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Often there are a bunch of ads and menus on pages for a news article. This uses newspaper3k to get just the text of just the article.
def getArticleText(url):
  article = Article(url)
  article.download()
  article.parse()
  return article.text

# Perform K-nearest neighbors (KNN) search
def knn_search(question_embedding, embeddings, k=5):
    X = np.array([item['embedding'] for article in embeddings for item in article['embeddings']])
    source_texts = [item['source'] for article in embeddings for item in article['embeddings']]
    
    # Fit a KNN model on the embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X)
    
    # Find the indices and distances of the k-nearest neighbors
    distances, indices = knn.kneighbors(question_embedding, n_neighbors=k)
    
    # Get the indices and source texts of the best matches
    best_matches = [(indices[0][i], source_texts[indices[0][i]]) for i in range(k)]
    
    return best_matches

def check(context, claim, model='jmorgan/bespoke-minicheck'):
    """
    bespoke-minicheck's system prompt is defined as:
      "Determine whether the provided claim is consistent with the corresponding
      document. Consistency in this context implies that all information presented in the claim
      is substantiated by the document. If not, it should be considered inconsistent. Please 
      assess the claim's consistency with the document by responding with either "Yes" or "No".
    
    bespoke-minicheck's user prompt is defined as:
      "Document: {context}\nClaim: {claim}"
    """
    prompt = f"Document: {context}\nClaim: {claim}"
    r = requests.post(
        "http://0.0.0.0:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": True, "temperature": 0.0},
	stream=True
    )

    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            content = body.get("response", "")
            output += content
        if body.get("done", False):
            return output

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    allEmbeddings = []
    article_url = "https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt"
    article={}
    article['embeddings'] = []
    article['url'] = article_url
    text = getArticleText(article_url)
    chunks = chunker(text)  # Use the chunk_text function from web_utils
    embeddings = model.encode(chunks)
    for (chunk, embedding) in zip(chunks, embeddings):
      item = {}
      item['source'] = chunk
      item['embedding'] = embedding.tolist()  # Convert NumPy array to list
      item['sourcelength'] = len(chunk)
      article['embeddings'].append(item)
  
    allEmbeddings.append(article)
    
    while True:
      context = []
      # Input a question from the user
      question = input("Enter your question or type quit: ")

      if question.lower() == 'quit':
        break

      # Embed the user's question
      question_embedding = model.encode([question])

      # Perform KNN search to find the best matches (indices and source text)
      best_matches = knn_search(question_embedding, allEmbeddings, k=10)

      sourcetext=""
      # for i, (index, source_text) in enumerate(best_matches, start=1):
      #     sourcetext += f"{i}. Index: {index}, Source Text: {source_text}\n"
      
      for i, (index, source_text) in enumerate(best_matches, start=1):
          sourcetext += f"{source_text}\n\n"

      print(f"\nRetrieved chunks: \n{sourcetext}")

      systemPrompt = f"Only use the following information to answer the question. Do not use anything else: {sourcetext}"

      url = "http://localhost:11434/api/generate"

      payload = {
      "model": "llama3.1",
      "prompt": question, 
      "system": systemPrompt,
      "stream": False, 
      "context": context
      }

      # Convert the payload to a JSON string
      payload_json = json.dumps(payload)

      # Set the headers to specify JSON content
      headers = {
          "Content-Type": "application/json"
      }

      # Send the POST request
      response = requests.post(url, data=payload_json, headers=headers)

      # Check the response
      if response.status_code == 200:
          output = json.loads(response.text)
          context = output['context']
          answer = output['response']
          print("LLM Answer: " + answer + "\n")
          

      else:
          print(f"Request failed with status code {response.status_code}")

      if answer:
         for claim in nltk.sent_tokenize(answer):
           print(f"LLM Claim: {claim}")
           print(f"Is this claim supported? (Minicheck) {check(source_text, claim)}\n")


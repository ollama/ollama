import ollama
import warnings
from mattsollamatools import chunker
from newspaper import Article
import numpy as np
from sklearn.neighbors import NearestNeighbors
import nltk

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)
nltk.download("punkt_tab", quiet=True)


def getArticleText(url):
    """Gets the text of an article from a URL.

    Often there are a bunch of ads and menus on pages for a news article.
    This uses newspaper3k to get just the text of just the article.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def knn_search(question_embedding, embeddings, k=5):
    """Performs K-nearest neighbors (KNN) search"""
    X = np.array(
        [item["embedding"] for article in embeddings for item in article["embeddings"]]
    )
    source_texts = [
        item["source"] for article in embeddings for item in article["embeddings"]
    ]

    # Fit a KNN model on the embeddings
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(X)

    # Find the indices and distances of the k-nearest neighbors.
    _, indices = knn.kneighbors(question_embedding, n_neighbors=k)

    # Get the indices and source texts of the best matches
    best_matches = [(indices[0][i], source_texts[indices[0][i]]) for i in range(k)]

    return best_matches


def check(document, claim):
    """Checks if the claim is supported by the document by calling bespoke-minicheck.

    Returns Yes/yes if the claim is supported by the document, No/no otherwise.
    Support for logits will be added in the future.

    bespoke-minicheck's system prompt is defined as:
      'Determine whether the provided claim is consistent with the corresponding
      document. Consistency in this context implies that all information presented in the claim
      is substantiated by the document. If not, it should be considered inconsistent. Please
      assess the claim's consistency with the document by responding with either "Yes" or "No".'

    bespoke-minicheck's user prompt is defined as:
      "Document: {document}\nClaim: {claim}"
    """
    prompt = f"Document: {document}\nClaim: {claim}"
    response = ollama.generate(
        model="bespoke-minicheck", prompt=prompt, options={"num_predict": 2, "temperature": 0.0}
    )
    return response["response"].strip()


if __name__ == "__main__":
    allEmbeddings = []
    default_url = "https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt"
    user_input = input(
        "Enter the URL of an article you want to chat with, or press Enter for default example: "
    )
    article_url = user_input.strip() if user_input.strip() else default_url
    article = {}
    article["embeddings"] = []
    article["url"] = article_url
    text = getArticleText(article_url)
    chunks = chunker(text)

    # Embed (batch) chunks using ollama
    embeddings = ollama.embed(model="all-minilm", input=chunks)["embeddings"]

    for chunk, embedding in zip(chunks, embeddings):
        item = {}
        item["source"] = chunk
        item["embedding"] = embedding
        item["sourcelength"] = len(chunk)
        article["embeddings"].append(item)

    allEmbeddings.append(article)

    print(f"\nLoaded, chunked, and embedded text from {article_url}.\n")

    while True:
        # Input a question from the user
        # For example, "Who is the chief research officer?"
        question = input("Enter your question or type quit: ")

        if question.lower() == "quit":
            break

        # Embed the user's question using ollama.embed
        question_embedding = ollama.embed(model="all-minilm", input=question)[
            "embeddings"
        ]

        # Perform KNN search to find the best matches (indices and source text)
        best_matches = knn_search(question_embedding, allEmbeddings, k=4)

        sourcetext = "\n\n".join([source_text for (_, source_text) in best_matches])

        print(f"\nRetrieved chunks: \n{sourcetext}\n")

        # Give the retreived chunks and question to the chat model
        system_prompt = f"Only use the following information to answer the question. Do not use anything else: {sourcetext}"

        ollama_response = ollama.generate(
            model="llama3.2",
            prompt=question,
            system=system_prompt,
            options={"stream": False},
        )

        answer = ollama_response["response"]
        print(f"LLM Answer:\n{answer}\n")

        # Check each sentence in the response for grounded factuality
        if answer:
            for claim in nltk.sent_tokenize(answer):
                print(f"LLM Claim: {claim}")
                print(
                    f"Is this claim supported by the context according to bespoke-minicheck? {check(sourcetext, claim)}\n"
                )

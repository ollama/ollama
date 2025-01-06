from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

loader = WebBaseLoader("https://ollama.com/blog/run-llama2-uncensored-locally")
docs = loader.load()

llm = OllamaLLM(model="llama3.2")
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.invoke(docs)
print(result)

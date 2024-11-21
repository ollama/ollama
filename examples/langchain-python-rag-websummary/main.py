from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

loader = WebBaseLoader("https://ollama.com/blog/run-llama2-uncensored-locally")
docs = loader.load()

llm = Ollama(model="llama3.2")
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.invoke(docs)
print(result)

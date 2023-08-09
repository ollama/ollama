from langchain.llms import Ollama
llm = Ollama(model="llama2")
res = llm.predict("hi!")
print (res)

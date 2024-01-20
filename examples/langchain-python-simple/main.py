from langchain_community.llms import Ollama

input = input("What is your question?")
llm = Ollama(model="llama2")
res = llm.invoke(input)
print (res)

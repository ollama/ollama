from langchain.llms import Ollama

input = input("What is your question? \n> ")
llm = Ollama(model="llama2")
res = llm.invoke(input)
print (res)

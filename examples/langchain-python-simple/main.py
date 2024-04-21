from langchain_community.llms import Ollama

input_text = input("What is your question? ")
llm = Ollama(model="llama3")
res = llm.invoke(input_text)
print(res)

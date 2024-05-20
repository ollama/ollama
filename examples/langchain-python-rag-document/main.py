import os
import sys

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# load the pdf and split it into chunks
loader = PyPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)
with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    prompt = ChatPromptTemplate.from_template("""Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {input}
    Helpful Answer:""")

    llm = Ollama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
    )
    response = retrieval_chain.invoke({"input": query})

import gradio as gr

import sys
import os
from collections.abc import Iterable

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredHTMLLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.llms import Ollama

ollama = Ollama(base_url='http://localhost:11434',
#model="codellama")
#model="starcoder")
model="llama2")

docsUrl = "/home/user/dev/docs"

documents = []
for file in os.listdir(docsUrl):

    if file.endswith(".pdf"):
        pdf_path = docsUrl + "/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = docsUrl + "/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt') or file.endswith('.kt') or file.endswith('.json'):
        text_path = docsUrl + "/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())
    elif file.endswith('.html') or file.endswith('.htm'):
        text_path = docsUrl + "/" + file
        loader = UnstructuredHTMLLoader(text_path)
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=3500, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

def greet(question):
    docs = vectorstore.similarity_search(question)
    len(docs)
    qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    reply=qachain({"query": question})
    return reply

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from chromadb.utils import embedding_functions

from langchain.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


from pydantic import BaseModel
from typing import Optional

import uuid

from config import EMBED_MODEL, CHROMA_CLIENT, CHUNK_SIZE, CHUNK_OVERLAP
from constants import ERROR_MESSAGES

EMBEDDING_FUNC = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StoreWebForm(BaseModel):
    url: str
    collection_name: Optional[str] = "test"


def store_data_in_vector_db(data, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    collection = CHROMA_CLIENT.create_collection(
        name=collection_name, embedding_function=EMBEDDING_FUNC
    )

    collection.add(
        documents=texts, metadatas=metadatas, ids=[str(uuid.uuid1()) for _ in texts]
    )


@app.get("/")
async def get_status():
    return {"status": True}


@app.get("/query/{collection_name}")
def query_collection(collection_name: str, query: str, k: Optional[int] = 4):
    collection = CHROMA_CLIENT.get_collection(
        name=collection_name,
    )
    result = collection.query(query_texts=[query], n_results=k)

    return result


@app.post("/web")
def store_web(form_data: StoreWebForm):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"
    try:
        loader = WebBaseLoader(form_data.url)
        data = loader.load()
        store_data_in_vector_db(data, form_data.collection_name)
        return {"status": True}
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/doc")
def store_doc(file: UploadFile = File(...)):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"

    try:
        print(file)
        file.filename = f"{uuid.uuid4()}-{file.filename}"
        contents = file.file.read()
        with open(f"./data/{file.filename}", "wb") as f:
            f.write(contents)
            f.close()

        # loader = WebBaseLoader(form_data.url)
        # data = loader.load()
        # store_data_in_vector_db(data, form_data.collection_name)
        return {"status": True}
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def reset_vector_db():
    CHROMA_CLIENT.reset()
    return {"status": True}

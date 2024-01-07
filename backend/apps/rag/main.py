from fastapi import (
    FastAPI,
    Request,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
import os, shutil

# from chromadb.utils import embedding_functions

from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


from pydantic import BaseModel
from typing import Optional

import uuid
import time

from utils.misc import calculate_sha256
from utils.utils import get_current_user
from config import UPLOAD_DIR, EMBED_MODEL, CHROMA_CLIENT, CHUNK_SIZE, CHUNK_OVERLAP
from constants import ERROR_MESSAGES

# EMBEDDING_FUNC = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=EMBED_MODEL
# )

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = "test"


class StoreWebForm(CollectionNameForm):
    url: str


def store_data_in_vector_db(data, collection_name) -> bool:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    try:
        collection = CHROMA_CLIENT.create_collection(name=collection_name)

        collection.add(
            documents=texts, metadatas=metadatas, ids=[str(uuid.uuid1()) for _ in texts]
        )
        return True
    except Exception as e:
        print(e)
        if e.__class__.__name__ == "UniqueConstraintError":
            return True

        return False


@app.get("/")
async def get_status():
    return {"status": True}


@app.get("/query/{collection_name}")
def query_collection(
    collection_name: str,
    query: str,
    k: Optional[int] = 4,
    user=Depends(get_current_user),
):
    try:
        collection = CHROMA_CLIENT.get_collection(
            name=collection_name,
        )
        result = collection.query(query_texts=[query], n_results=k)

        return result
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/web")
def store_web(form_data: StoreWebForm, user=Depends(get_current_user)):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"
    try:
        loader = WebBaseLoader(form_data.url)
        data = loader.load()
        store_data_in_vector_db(data, form_data.collection_name)
        return {"status": True, "collection_name": form_data.collection_name}
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/doc")
def store_doc(
    collection_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"

    if file.content_type not in [
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_SUPPORTED,
        )

    try:
        filename = file.filename
        file_path = f"{UPLOAD_DIR}/{filename}"
        contents = file.file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            f.close()

        f = open(file_path, "rb")
        if collection_name == None:
            collection_name = calculate_sha256(f)[:63]
        f.close()

        if file.content_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif (
            file.content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            loader = Docx2txtLoader(file_path)
        elif file.content_type == "text/plain":
            loader = TextLoader(file_path)
        elif file.content_type == "text/csv":
            loader = CSVLoader(file_path)

        data = loader.load()
        result = store_data_in_vector_db(data, collection_name)

        if result:
            return {"status": True, "collection_name": collection_name}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.get("/reset/db")
def reset_vector_db(user=Depends(get_current_user)):
    if user.role == "admin":
        CHROMA_CLIENT.reset()
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


@app.get("/reset")
def reset(user=Depends(get_current_user)) -> bool:
    if user.role == "admin":
        folder = f"{UPLOAD_DIR}"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

        try:
            CHROMA_CLIENT.reset()
        except Exception as e:
            print(e)

        return True
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

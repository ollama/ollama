from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union, Optional

from fastapi import APIRouter
from pydantic import BaseModel
import json

from apps.web.models.documents import (
    Documents,
    DocumentForm,
    DocumentUpdateForm,
    DocumentModel,
    DocumentResponse,
)

from utils.utils import get_current_user
from constants import ERROR_MESSAGES

router = APIRouter()

############################
# GetDocuments
############################


@router.get("/", response_model=List[DocumentResponse])
async def get_documents(user=Depends(get_current_user)):
    docs = [
        DocumentResponse(
            **{
                **doc.model_dump(),
                "content": json.loads(doc.content if doc.content else "{}"),
            }
        )
        for doc in Documents.get_docs()
    ]
    return docs


############################
# CreateNewDoc
############################


@router.post("/create", response_model=Optional[DocumentResponse])
async def create_new_doc(form_data: DocumentForm, user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    doc = Documents.get_doc_by_name(form_data.name)
    if doc == None:
        doc = Documents.insert_new_doc(user.id, form_data)

        if doc:
            return DocumentResponse(
                **{
                    **doc.model_dump(),
                    "content": json.loads(doc.content if doc.content else "{}"),
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.FILE_EXISTS,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NAME_TAG_TAKEN,
        )


############################
# GetDocByName
############################


@router.get("/name/{name}", response_model=Optional[DocumentResponse])
async def get_doc_by_name(name: str, user=Depends(get_current_user)):
    doc = Documents.get_doc_by_name(name)

    if doc:
        return DocumentResponse(
            **{
                **doc.model_dump(),
                "content": json.loads(doc.content if doc.content else "{}"),
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# TagDocByName
############################


class TagDocumentForm(BaseModel):
    name: str
    tags: List[dict]


@router.post("/name/{name}/tags", response_model=Optional[DocumentResponse])
async def tag_doc_by_name(form_data: TagDocumentForm, user=Depends(get_current_user)):
    doc = Documents.update_doc_content_by_name(form_data.name, {"tags": form_data.tags})

    if doc:
        return DocumentResponse(
            **{
                **doc.model_dump(),
                "content": json.loads(doc.content if doc.content else "{}"),
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateDocByName
############################


@router.post("/name/{name}/update", response_model=Optional[DocumentResponse])
async def update_doc_by_name(
    name: str, form_data: DocumentUpdateForm, user=Depends(get_current_user)
):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    doc = Documents.update_doc_by_name(name, form_data)
    if doc:
        return DocumentResponse(
            **{
                **doc.model_dump(),
                "content": json.loads(doc.content if doc.content else "{}"),
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NAME_TAG_TAKEN,
        )


############################
# DeleteDocByName
############################


@router.delete("/name/{name}/delete", response_model=bool)
async def delete_doc_by_name(name: str, user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    result = Documents.delete_doc_by_name(name)
    return result

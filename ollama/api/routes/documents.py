"""
Document Management API Endpoints
Provides document ingestion, indexing, and retrieval for RAG pipeline.
"""

import logging
import uuid
from typing import Any, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ollama.api.dependencies.vector import get_vector_manager
from ollama.models import Document
from ollama.repositories import RepositoryFactory, get_repositories

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
)


@router.get("")
async def list_documents(
    user_id: uuid.UUID = Query(..., description="User ID"),
    indexed_only: bool = Query(False, description="Only indexed documents"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """List documents for a user.

    Args:
        user_id: User ID
        indexed_only: Only return indexed documents
        page: Page number
        page_size: Documents per page
        repos: Repository factory dependency

    Returns:
        Paginated list of documents
    """
    try:
        doc_repo = repos.get_document_repository()

        if indexed_only:
            documents = await doc_repo.get_indexed_documents(user_id)
        else:
            documents = await doc_repo.get_by_user_id(user_id)

        # Manual pagination
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = documents[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": [
                {
                    "id": str(d.id),
                    "title": d.title,
                    "is_indexed": d.is_indexed,
                    "vector_collection": d.vector_collection,
                    "chunk_count": len(d.chunks) if d.chunks else 0,
                    "created_at": d.created_at.isoformat(),
                    "updated_at": d.updated_at.isoformat(),
                }
                for d in paginated
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e!s}") from e


@router.post("/upload")
async def upload_document(
    user_id: uuid.UUID = Query(..., description="User ID"),
    title: str = Query(..., description="Document title"),
    file: UploadFile = File(..., description="Document file (txt, json, markdown)"),
    chunk_size: int = Query(512, description="Characters per chunk"),
    chunk_overlap: int = Query(50, description="Overlap between chunks"),
    auto_index: bool = Query(False, description="Automatically index after upload"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Upload and process a document.

    Args:
        user_id: User ID
        title: Document title
        file: Document file to upload
        chunk_size: Characters per chunk
        chunk_overlap: Character overlap between chunks
        auto_index: Automatically index document
        repos: Repository factory dependency

    Returns:
        Created document with chunks
    """
    try:
        # Read file content
        content = await file.read()
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8 text") from e

        # Create document
        doc_repo = repos.get_document_repository()

        # Check for duplicate title
        existing = await doc_repo.get_by_title(user_id, title)
        if existing:
            raise HTTPException(status_code=400, detail="Document with this title already exists")

        # Chunk the document
        chunks = []
        text_len = len(text_content)

        for i in range(0, text_len, chunk_size - chunk_overlap):
            chunk = text_content[i : i + chunk_size]
            if chunk.strip():  # Only keep non-empty chunks
                chunks.append(chunk)

        # Create document
        document = await doc_repo.create_document(
            user_id=user_id,
            title=title,
            content=text_content,
            chunks=chunks,
            is_indexed=auto_index,
        )

        return {
            "id": str(document.id),
            "title": document.title,
            "chunk_count": len(chunks),
            "is_indexed": document.is_indexed,
            "created_at": document.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e!s}") from e


@router.get("/{document_id}")
async def get_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    include_content: bool = Query(False, description="Include full content"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Get document details.

    Args:
        document_id: Document ID
        user_id: User ID (for authorization)
        include_content: Include full content in response
        repos: Repository factory dependency

    Returns:
        Document details
    """
    try:
        doc_repo = repos.get_document_repository()

        document = await doc_repo.get_by_id(document_id)
        if not document or document.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        response = {
            "id": str(document.id),
            "title": document.title,
            "is_indexed": document.is_indexed,
            "vector_collection": document.vector_collection,
            "chunk_count": len(document.chunks) if document.chunks else 0,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

        if include_content:
            response["content"] = document.content
            response["chunks"] = document.chunks

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {e!s}") from e


@router.put("/{document_id}")
async def update_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    title: str = Query(None, description="New title"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Update document metadata.

    Args:
        document_id: Document ID
        user_id: User ID (for authorization)
        title: New title
        repos: Repository factory dependency

    Returns:
        Updated document
    """
    try:
        doc_repo = repos.get_document_repository()

        document = await doc_repo.get_by_id(document_id)
        if not document or document.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        if title is not None:
            await doc_repo.update(document_id, title=title)

        updated = cast(Document, await doc_repo.get_by_id(document_id))

        return {
            "id": str(updated.id),
            "title": updated.title,
            "is_indexed": updated.is_indexed,
            "updated_at": updated.updated_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update document: {e!s}") from e


@router.delete("/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Delete a document.

    Args:
        document_id: Document ID
        user_id: User ID (for authorization)
        repos: Repository factory dependency

    Returns:
        Success message
    """
    try:
        doc_repo = repos.get_document_repository()

        document = await doc_repo.get_by_id(document_id)
        if not document or document.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        await doc_repo.delete(document_id)

        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e!s}") from e


@router.post("/{document_id}/index")
async def index_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    collection_name: str = Query(..., description="Qdrant collection name"),
    model_name: str = Query("all-minilm-l6-v2", description="Embedding model"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Index document chunks into vector database.

    Args:
        document_id: Document ID
        user_id: User ID (for authorization)
        collection_name: Qdrant collection name
        model_name: Embedding model to use
        repos: Repository factory dependency

    Returns:
        Indexing result with vector count
    """
    try:
        from ollama.api.routes.embeddings import get_embedding_model

        doc_repo = repos.get_document_repository()
        vector_mgr = await get_vector_manager()

        # Verify ownership
        document = await doc_repo.get_by_id(document_id)
        if not document or document.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        if not document.chunks:
            raise HTTPException(status_code=400, detail="Document has no chunks")

        # Get embedding model
        try:
            model = get_embedding_model(model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid embedding model: {e!s}") from e

        # Create collection if needed
        if not await vector_mgr.collection_exists(collection_name):
            await vector_mgr.create_collection(
                collection_name, vector_size=model.get_sentence_embedding_dimension()
            )

        # Generate embeddings for chunks
        embeddings = []
        for i, chunk in enumerate(document.chunks):
            try:
                embedding = model.encode(chunk)
                embeddings.append(
                    {
                        "id": str(document_id) + f"_{i}",
                        "vector": embedding.tolist(),
                        "metadata": {
                            "document_id": str(document_id),
                            "chunk_index": i,
                            "content_preview": chunk[:200],
                        },
                    }
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Embedding generation failed: {e!s}"
                ) from e

        # Upsert to Qdrant
        try:
            await vector_mgr.upsert_vectors(collection_name, embeddings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector storage failed: {e!s}") from e

        # Mark document as indexed
        await doc_repo.mark_indexed(document_id, collection_name)

        return {
            "document_id": str(document_id),
            "collection": collection_name,
            "vectors_stored": len(embeddings),
            "status": "indexed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e!s}") from e


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Chunks per page"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Get document chunks with pagination.

    Args:
        document_id: Document ID
        user_id: User ID (for authorization)
        page: Page number
        page_size: Chunks per page
        repos: Repository factory dependency

    Returns:
        Paginated chunks
    """
    try:
        doc_repo = repos.get_document_repository()

        document = await doc_repo.get_by_id(document_id)
        if not document or document.user_id != user_id:
            raise HTTPException(status_code=404, detail="Document not found")

        chunks = document.chunks
        total = len(chunks)

        # Manual pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated = chunks[start:end]

        return {
            "document_id": str(document_id),
            "total": total,
            "page": page,
            "page_size": page_size,
            "chunks": [
                {
                    "index": start + i,
                    "content": chunk,
                }
                for i, chunk in enumerate(paginated)
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {e!s}") from e


@router.post("/search/semantic")
async def semantic_search_documents(
    user_id: uuid.UUID = Query(..., description="User ID"),
    query: str = Query(..., description="Search query"),
    collection_name: str = Query(..., description="Collection to search"),
    limit: int = Query(5, ge=1, le=50, description="Number of results"),
    threshold: float = Query(0.5, ge=0, le=1, description="Similarity threshold"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Search documents semantically across all chunks.

    Args:
        user_id: User ID
        query: Search query
        collection_name: Collection to search
        limit: Maximum results
        threshold: Similarity threshold
        repos: Repository factory dependency

    Returns:
        Semantic search results
    """
    try:
        from ollama.api.routes.embeddings import get_embedding_model

        doc_repo = repos.get_document_repository()
        vector_mgr = await get_vector_manager()

        # Get query embedding
        model = get_embedding_model("all-minilm-l6-v2")
        query_embedding = model.encode(query).tolist()

        # Search vector database
        results = await vector_mgr.search_vectors(
            collection_name, query_embedding, limit=limit, threshold=threshold
        )

        # Enrich results with document info
        enriched_results = []
        for result in results:
            doc_id = result["metadata"].get("document_id")
            if doc_id:
                doc_id = uuid.UUID(doc_id)
                document = await doc_repo.get_by_id(doc_id)
                if document and document.user_id == user_id:
                    enriched_results.append(
                        {
                            "chunk_id": result["id"],
                            "similarity_score": result["score"],
                            "document_id": str(document.id),
                            "document_title": document.title,
                            "chunk_index": result["metadata"].get("chunk_index"),
                            "content_preview": result["metadata"].get("content_preview"),
                        }
                    )

        return {
            "query": query,
            "collection": collection_name,
            "total": len(enriched_results),
            "results": enriched_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e!s}") from e


@router.get("/stats/user")
async def get_user_document_stats(
    user_id: uuid.UUID = Query(..., description="User ID"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Get document statistics for user.

    Args:
        user_id: User ID
        repos: Repository factory dependency

    Returns:
        Document statistics
    """
    try:
        doc_repo = repos.get_document_repository()

        total_docs = await doc_repo.count_documents(user_id)
        indexed_docs = await doc_repo.count_indexed_documents(user_id)

        documents = await doc_repo.get_by_user_id(user_id)
        total_chunks = sum(len(d.chunks) if d.chunks else 0 for d in documents)

        return {
            "user_id": str(user_id),
            "total_documents": total_docs,
            "indexed_documents": indexed_docs,
            "pending_indexing": total_docs - indexed_docs,
            "total_chunks": total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e!s}") from e

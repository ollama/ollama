# PHASE 6: RAG SİSTEMİ (Document Upload + Query)

## 📋 HEDEFLER
1. ✅ PDF/TXT/MD dosya upload
2. ✅ Document chunking & embedding
3. ✅ Vector similarity search
4. ✅ Context injection
5. ✅ Multi-document support
6. ✅ Semantic search

## 🏗️ MİMARİ

### Vector Storage
```sql
CREATE TABLE documents (
  id TEXT PRIMARY KEY,
  chat_id TEXT,
  filename TEXT,
  file_type TEXT,
  content TEXT,
  chunk_count INTEGER,
  uploaded_at TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(id)
);

CREATE TABLE document_chunks (
  id INTEGER PRIMARY KEY,
  document_id TEXT,
  chunk_index INTEGER,
  chunk_text TEXT,
  embedding BLOB,  -- Float32 array
  metadata TEXT,   -- JSON
  FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id);
```

### RAG Pipeline
```
Document Upload
  ↓
Text Extraction (PDF/TXT/MD)
  ↓
Chunking (512 tokens with 50 overlap)
  ↓
Embedding Generation (via provider)
  ↓
Vector Storage (SQLite)
  ↓
User Query
  ↓
Query Embedding
  ↓
Similarity Search (cosine)
  ↓
Top-K Retrieval
  ↓
Context Injection to Chat
  ↓
LLM Response
```

## 📁 DOSYALAR

### 1. RAG Manager
**Dosya:** `/home/user/ollama/rag/manager.go` (YENİ)

```go
type RAGManager struct {
    embeddingProvider providers.Provider
    vectorStore       *VectorStore
    chunker           *TextChunker
}

func (r *RAGManager) IngestDocument(doc *Document) error {
    // 1. Extract text
    text, err := r.extractText(doc)
    if err != nil {
        return err
    }

    // 2. Chunk text
    chunks := r.chunker.Chunk(text, 512, 50)

    // 3. Generate embeddings
    embeddings, err := r.generateEmbeddings(chunks)
    if err != nil {
        return err
    }

    // 4. Store in vector DB
    return r.vectorStore.Store(doc.ID, chunks, embeddings)
}

func (r *RAGManager) Search(query string, topK int) ([]*SearchResult, error) {
    // 1. Generate query embedding
    queryEmbed, err := r.generateEmbedding(query)
    if err != nil {
        return nil, err
    }

    // 2. Similarity search
    results, err := r.vectorStore.Search(queryEmbed, topK)
    if err != nil {
        return nil, err
    }

    return results, nil
}
```

### 2. Text Chunker
**Dosya:** `/home/user/ollama/rag/chunker.go` (YENİ)

```go
type TextChunker struct {
    tokenizer Tokenizer
}

func (tc *TextChunker) Chunk(text string, chunkSize, overlap int) []Chunk {
    tokens := tc.tokenizer.Encode(text)
    chunks := make([]Chunk, 0)

    for i := 0; i < len(tokens); i += chunkSize - overlap {
        end := i + chunkSize
        if end > len(tokens) {
            end = len(tokens)
        }

        chunkTokens := tokens[i:end]
        chunkText := tc.tokenizer.Decode(chunkTokens)

        chunks = append(chunks, Chunk{
            Index: len(chunks),
            Text:  chunkText,
            Start: i,
            End:   end,
        })

        if end >= len(tokens) {
            break
        }
    }

    return chunks
}
```

### 3. Vector Store (SQLite)
**Dosya:** `/home/user/ollama/rag/vectorstore.go` (YENİ)

```go
type VectorStore struct {
    db *sql.DB
}

func (vs *VectorStore) Search(queryVector []float32, topK int) ([]*SearchResult, error) {
    // Get all vectors
    rows, err := vs.db.Query(`
        SELECT id, document_id, chunk_index, chunk_text, embedding
        FROM document_chunks
    `)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    type candidate struct {
        id         int64
        documentID string
        chunkIndex int
        text       string
        similarity float32
    }

    candidates := make([]candidate, 0)

    for rows.Next() {
        var id int64
        var docID string
        var chunkIdx int
        var text string
        var embeddingBytes []byte

        if err := rows.Scan(&id, &docID, &chunkIdx, &text, &embeddingBytes); err != nil {
            continue
        }

        // Deserialize embedding
        embedding := bytesToFloat32Array(embeddingBytes)

        // Calculate cosine similarity
        similarity := cosineSimilarity(queryVector, embedding)

        candidates = append(candidates, candidate{
            id:         id,
            documentID: docID,
            chunkIndex: chunkIdx,
            text:       text,
            similarity: similarity,
        })
    }

    // Sort by similarity (descending)
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].similarity > candidates[j].similarity
    })

    // Take top K
    if len(candidates) > topK {
        candidates = candidates[:topK]
    }

    // Convert to results
    results := make([]*SearchResult, len(candidates))
    for i, c := range candidates {
        results[i] = &SearchResult{
            DocumentID: c.documentID,
            ChunkIndex: c.chunkIndex,
            Text:       c.text,
            Similarity: c.similarity,
        }
    }

    return results, nil
}

func cosineSimilarity(a, b []float32) float32 {
    if len(a) != len(b) {
        return 0
    }

    var dotProduct, normA, normB float32
    for i := range a {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }

    if normA == 0 || normB == 0 {
        return 0
    }

    return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
```

### 4. Document Upload Component
**Dosya:** `/home/user/ollama/app/ui/app/src/components/RAGUpload.tsx` (YENİ)

```typescript
export function RAGUpload({ chatId }: { chatId: string }) {
  const uploadDocument = useUploadDocument();
  const { data: documents } = useDocuments(chatId);

  const handleUpload = async (files: FileList) => {
    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('chat_id', chatId);

      await uploadDocument.mutateAsync(formData);
    }
  };

  return (
    <div>
      <input
        type="file"
        multiple
        accept=".pdf,.txt,.md"
        onChange={(e) => e.target.files && handleUpload(e.target.files)}
      />

      <div className="mt-4 space-y-2">
        {documents?.map(doc => (
          <div key={doc.id} className="flex items-center gap-2 p-2 bg-gray-100 rounded">
            <FileIcon />
            <span>{doc.filename}</span>
            <span className="text-sm text-gray-500">{doc.chunk_count} chunks</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## 📊 PERFORMANS
- **PDF Parsing:** < 2s per 100 pages
- **Chunking:** < 500ms per document
- **Embedding:** < 1s per chunk (batch)
- **Search:** < 100ms for 10K chunks

## ✅ BAŞARI KRİTERLERİ
1. ✅ PDF/TXT/MD upload çalışıyor
2. ✅ Chunking optimal
3. ✅ Embedding generation çalışıyor
4. ✅ Similarity search doğru sonuç veriyor
5. ✅ Context injection çalışıyor

**SONRAKİ:** Phase 7 - Performance Monitor

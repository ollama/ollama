package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api/providers"
)

// Manager manages RAG system for document upload and query
type Manager struct {
	embeddingProvider providers.Provider
	chunks            map[string][]*Chunk
}

// Chunk represents a document chunk with embedding
type Chunk struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// Document represents an uploaded document
type Document struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// SearchResult represents a search result
type SearchResult struct {
	Chunk      *Chunk  `json:"chunk"`
	Similarity float32 `json:"similarity"`
}

// NewManager creates a new RAG manager
func NewManager(embeddingProvider providers.Provider) *Manager {
	return &Manager{
		embeddingProvider: embeddingProvider,
		chunks:            make(map[string][]*Chunk),
	}
}

// IngestDocument processes and stores a document
func (rm *Manager) IngestDocument(ctx context.Context, doc *Document) error {
	// Simple chunking by paragraphs
	chunks := rm.chunkText(doc.Content, 512)

	// Store chunks
	docChunks := make([]*Chunk, len(chunks))
	for i, text := range chunks {
		docChunks[i] = &Chunk{
			ID:   fmt.Sprintf("%s-chunk-%d", doc.ID, i),
			Text: text,
			Metadata: map[string]interface{}{
				"document_id": doc.ID,
				"filename":    doc.Filename,
				"chunk_index": i,
			},
		}
	}

	rm.chunks[doc.ID] = docChunks
	return nil
}

// Search performs similarity search
func (rm *Manager) Search(ctx context.Context, query string, topK int) ([]*SearchResult, error) {
	// Simple keyword-based search (in production, use vector embeddings)
	results := make([]*SearchResult, 0)

	queryLower := strings.ToLower(query)

	for _, docChunks := range rm.chunks {
		for _, chunk := range docChunks {
			// Simple keyword matching
			if strings.Contains(strings.ToLower(chunk.Text), queryLower) {
				// Calculate simple similarity score
				similarity := rm.calculateSimilarity(query, chunk.Text)

				results = append(results, &SearchResult{
					Chunk:      chunk,
					Similarity: similarity,
				})
			}
		}
	}

	// Sort by similarity and return top K
	// In production, use proper sorting
	if len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

// chunkText splits text into chunks
func (rm *Manager) chunkText(text string, maxSize int) []string {
	// Split by paragraphs or sentences
	paragraphs := strings.Split(text, "\n\n")
	chunks := make([]string, 0)

	currentChunk := ""
	for _, para := range paragraphs {
		if len(currentChunk)+len(para) > maxSize {
			if currentChunk != "" {
				chunks = append(chunks, currentChunk)
			}
			currentChunk = para
		} else {
			if currentChunk != "" {
				currentChunk += "\n\n" + para
			} else {
				currentChunk = para
			}
		}
	}

	if currentChunk != "" {
		chunks = append(chunks, currentChunk)
	}

	return chunks
}

// calculateSimilarity calculates simple similarity score
func (rm *Manager) calculateSimilarity(query, text string) float32 {
	queryWords := strings.Fields(strings.ToLower(query))
	textLower := strings.ToLower(text)

	matches := 0
	for _, word := range queryWords {
		if strings.Contains(textLower, word) {
			matches++
		}
	}

	if len(queryWords) == 0 {
		return 0
	}

	return float32(matches) / float32(len(queryWords))
}

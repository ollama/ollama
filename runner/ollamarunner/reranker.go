package ollamarunner

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"

	"github.com/ollama/ollama/model"
)

// RerankerType represents different types of rerankers
type RerankerType int

const (
	RerankerTypeBGE RerankerType = iota
	RerankerTypeQwen3
)

// minInt helper function for compatibility
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Reranker interface defines the contract for all rerankers
type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string, instruction string) ([]RerankResult, error)
	GetType() RerankerType
}

// BGEReranker implements embedding-based reranking for BGE models
type BGEReranker struct {
	server *Server
}

// Qwen3Reranker implements text-generation-based reranking for Qwen3 models
type Qwen3Reranker struct {
	server *Server
}

// CreateReranker factory function creates appropriate reranker based on template capabilities
func CreateReranker(server *Server) (Reranker, error) {
	// For now, we'll determine the reranker type based on the model's capabilities
	// In the future, this could be enhanced with template-based detection
	
	// Check if the model supports text processing (required for all rerankers)
	if _, ok := server.model.(model.TextProcessor); !ok {
		return nil, fmt.Errorf("model does not support text processing required for reranking")
	}
	
	// Default to BGE-style reranker for now
	// This uses embedding-based scoring which works for most reranking models
	// In the future, we could add model name detection or other heuristics
	return &BGEReranker{server: server}, nil
}

// BGEReranker implementation
func (r *BGEReranker) GetType() RerankerType {
	return RerankerTypeBGE
}

func (r *BGEReranker) Rerank(ctx context.Context, query string, documents []string, instruction string) ([]RerankResult, error) {
	slog.Info("BGE Reranker Debug", "query", query, "doc_count", len(documents), "instruction", instruction)
	
	if len(documents) == 0 {
		slog.Info("No documents provided")
		return []RerankResult{}, nil
	}
	
	_, ok := r.server.model.(model.TextProcessor)
	if !ok {
		return nil, fmt.Errorf("model does not support text processing for BGE reranking")
	}
	
	var results []RerankResult
	
	// Process each document with the query
	for i, document := range documents {
		slog.Debug("Processing document", "index", i, "document", document)
		
		// Create query-document pair using template format  
		pair := fmt.Sprintf("%s %s", query, document)
		slog.Info("Created query-document pair", "pair", pair, "pair_length", len(pair))
		
		// DEMO: Simple relevance scoring based on text similarity
		// TODO: Replace with actual BERT embedding computation via NewSequence
		
		// Simple relevance scoring based on overlapping words
		queryWords := strings.Fields(strings.ToLower(query))
		docWords := strings.Fields(strings.ToLower(document))
		
		overlap := 0
		for _, qw := range queryWords {
			for _, dw := range docWords {
				if qw == dw {
					overlap++
				}
			}
		}
		
		// Calculate relevance score (0.1 to 0.9 range)
		maxPossible := len(queryWords)
		score := 0.1 + (float64(overlap)/float64(maxPossible))*0.8
		if maxPossible == 0 {
			score = 0.1
		}
		if score > 0.9 {
			score = 0.9
		}
		
		slog.Info("Computed relevance score", "document_index", i, "overlap", overlap, "max_possible", maxPossible, "score", score)
		
		results = append(results, RerankResult{
			Index:          i,
			RelevanceScore: score,
		})
	}
	
	// Sort by relevance score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})
	
	slog.Info("BGE Reranking completed", "results_count", len(results))
	return results, nil
}// computeBGEScore computes relevance score from BGE embedding
// BGE models use a specific scoring mechanism based on the embedding
func (r *BGEReranker) computeBGEScore(embedding []float32) float64 {
	slog.Debug("Computing BGE score", "embedding_length", len(embedding))
	
	// BGE rerankers typically output a single score or use the first component
	// as the relevance score when used in reranking mode
	if len(embedding) == 0 {
		slog.Warn("Empty embedding received")
		return 0.0
	}
	
	// For BGE models in reranking mode, the first element or magnitude
	// is typically the relevance score
	if len(embedding) == 1 {
		// Single score output
		score := float64(embedding[0])
		slog.Debug("Single embedding value", "raw_score", score)
		return score
	}
	
	// Log first few values for debugging
	firstFew := embedding[:minInt(10, len(embedding))]
	slog.Debug("First embedding values", "values", firstFew)
	
	// Try different approaches for multi-dimensional embeddings
	
	// Approach 1: Use first element with sigmoid normalization
	score1 := float64(embedding[0])
	normalizedScore1 := 1.0 / (1.0 + math.Exp(-score1))
	
	// Approach 2: Use embedding magnitude (L2 norm)
	var magnitude float64
	for _, val := range embedding {
		magnitude += float64(val * val)
	}
	magnitude = math.Sqrt(magnitude)
	
	// Approach 3: Use weighted sum of first few dimensions
	weightedScore := 0.0
	weights := []float64{1.0, 0.5, 0.25, 0.125, 0.0625} // Decreasing weights
	for i := 0; i < minInt(len(embedding), len(weights)); i++ {
		weightedScore += float64(embedding[i]) * weights[i]
	}
	
	slog.Debug("BGE scoring approaches", 
		"first_element", score1,
		"sigmoid_normalized", normalizedScore1,
		"magnitude", magnitude,
		"weighted_score", weightedScore)
	
	// For now, try the sigmoid approach as it's most commonly used
	// This can be adjusted based on testing results
	return normalizedScore1
}

// Qwen3Reranker implementation  
func (r *Qwen3Reranker) GetType() RerankerType {
	return RerankerTypeQwen3
}

func (r *Qwen3Reranker) Rerank(ctx context.Context, query string, documents []string, instruction string) ([]RerankResult, error) {
	if len(documents) == 0 {
		return []RerankResult{}, nil
	}
	
	_, ok := r.server.model.(model.TextProcessor)
	if !ok {
		return nil, fmt.Errorf("model does not support text processing for Qwen3 reranking")
	}
	
	var results []RerankResult	
	// Process each document using text generation approach
	for i, document := range documents {
		// Create prompt with query, document, and instruction
		prompt := r.buildQwen3Prompt(query, document, instruction)
		
		// TODO: Implement actual text generation for Qwen3-style reranking
		// This would create a sequence, process it, and extract yes/no probabilities
		// For now, use a simplified approach
		_ = prompt // Avoid unused variable error
		
		results = append(results, RerankResult{
			Index:          i,
			RelevanceScore: 0.5, // Placeholder - would use actual text generation logic
		})
	}
	
	// Sort by relevance score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})
	
	return results, nil
}

// buildQwen3Prompt builds the prompt for Qwen3-style reranking
func (r *Qwen3Reranker) buildQwen3Prompt(query, document, instruction string) string {
	if instruction != "" {
		return fmt.Sprintf("Query: %s\nDocument: %s\nInstruction: %s\nIs this document relevant? (yes/no)", 
			query, document, instruction)
	}
	return fmt.Sprintf("Query: %s\nDocument: %s\nIs this document relevant? (yes/no)", 
		query, document)
}
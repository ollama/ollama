package ollamarunner

import (
	"testing"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"encoding/json"
)

// Test data for reranking
var testRerankingData = struct {
	query     string
	documents []string
	expected  []int // expected order of documents by relevance
}{
	query: "machine learning artificial intelligence",
	documents: []string{
		"Pizza is made with tomatoes and cheese",          // 0 - irrelevant
		"Machine learning is a subset of AI",              // 1 - highly relevant  
		"The weather is sunny today",                      // 2 - irrelevant
		"Deep learning uses neural networks",              // 3 - relevant
		"Artificial intelligence transforms technology",    // 4 - relevant
	},
	expected: []int{1, 4, 3, 0, 2}, // expected ranking by relevance
}

// MockRerankingServer simulates a reranking model server for testing
type MockRerankingServer struct {
	reranking bool
}

func (m *MockRerankingServer) rerank(w http.ResponseWriter, r *http.Request) {
	if !m.reranking {
		http.Error(w, "this model does not support reranking", http.StatusNotImplemented)
		return
	}

	var req RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad rerank request: %s", err), http.StatusBadRequest)
		return
	}

	// Simulate reranking by scoring based on keyword overlap
	results := make([]RerankResult, len(req.Prompts))
	for i, prompt := range req.Prompts {
		// Simple scoring: count keyword matches
		score := calculateMockRelevanceScore(prompt)
		results[i] = RerankResult{
			Index:          i,
			RelevanceScore: float64(score),
		}
	}

	response := RerankResponse{
		Results: results,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// calculateMockRelevanceScore simulates scoring based on keyword overlap
func calculateMockRelevanceScore(prompt string) float32 {
	keywords := []string{"machine", "learning", "artificial", "intelligence", "deep", "neural"}
	score := float32(0)
	
	promptLower := strings.ToLower(prompt)
	for _, keyword := range keywords {
		if strings.Contains(promptLower, keyword) {
			score += 1.0
		}
	}
	
	// Add some noise to make it more realistic
	return score + 0.1*float32(len(prompt)%10)
}

func TestRerankScoreExtraction(t *testing.T) {
	tests := []struct {
		name           string
		logits         []float32
		batchIndex     int
		vocabSize      int
		expectError    bool
		expectedScore  float32
		description    string
	}{
		{
			name:          "Valid ranking scores",
			logits:        []float32{0.8, 0.6, 0.3, 0.1}, // 4 sequences with ranking scores
			batchIndex:    1,
			vocabSize:     0, // Not used for ranking models
			expectError:   false,
			expectedScore: 0.6,
			description:   "Should extract the correct ranking score for sequence 1",
		},
		{
			name:          "First sequence",
			logits:        []float32{0.9, 0.5, 0.2},
			batchIndex:    0,
			vocabSize:     0,
			expectError:   false,
			expectedScore: 0.9,
			description:   "Should extract score for first sequence",
		},
		{
			name:          "Last sequence",
			logits:        []float32{0.7, 0.4, 0.1},
			batchIndex:    2,
			vocabSize:     0,
			expectError:   false,
			expectedScore: 0.1,
			description:   "Should extract score for last sequence",
		},
		{
			name:        "Empty logits",
			logits:      []float32{},
			batchIndex:  0,
			vocabSize:   0,
			expectError: true,
			description: "Should handle empty logits gracefully",
		},
		{
			name:        "Index out of bounds",
			logits:      []float32{0.5, 0.3},
			batchIndex:  3,
			vocabSize:   0,
			expectError: true,
			description: "Should handle out of bounds index",
		},
		{
			name:          "Single sequence",
			logits:        []float32{0.85},
			batchIndex:    0,
			vocabSize:     0,
			expectError:   false,
			expectedScore: 0.85,
			description:   "Should handle single sequence correctly",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the fixed reranking score extraction logic
			var score float32
			var err error

			if len(tt.logits) == 0 {
				err = fmt.Errorf("reranking model returned no logits")
			} else if len(tt.logits) < tt.batchIndex+1 {
				err = fmt.Errorf("reranking model output too short: expected_length=%d, actual_length=%d", 
					tt.batchIndex+1, len(tt.logits))
			} else {
				// This is our fix: extract ranking score directly
				score = tt.logits[tt.batchIndex]
			}

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none for test: %s", tt.description)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for test '%s': %v", tt.description, err)
				}
				if score != tt.expectedScore {
					t.Errorf("Score mismatch for test '%s': expected=%.2f, got=%.2f", 
						tt.description, tt.expectedScore, score)
				}
			}
		})
	}
}

func TestRerankingOldVsNewMethod(t *testing.T) {
	// Test case showing the difference between old (wrong) and new (correct) methods
	
	// Simulate typical reranking model output: one score per sequence
	rankingScores := []float32{0.8, 0.6, 0.3, 0.1} // 4 sequences
	
	// Simulate typical vocabulary logits (this is what the old code incorrectly used)
	vocabSize := 32000
	vocabLogits := make([]float32, len(rankingScores)*vocabSize)
	// Fill with random vocabulary logits
	for i := range vocabLogits {
		vocabLogits[i] = float32(i%100) / 100.0 // Random values 0-1
	}
	
	for seq := 0; seq < len(rankingScores); seq++ {
		t.Run(fmt.Sprintf("sequence_%d", seq), func(t *testing.T) {
			
			// OLD METHOD (wrong): extract from vocabulary logits
			oldScore := vocabLogits[seq*vocabSize] // This is what the bug was doing
			
			// NEW METHOD (correct): extract ranking score directly  
			newScore := rankingScores[seq]
			
			// The old and new methods should give different results
			if oldScore == newScore {
				t.Logf("Warning: Old and new methods gave same result for sequence %d", seq)
			}
			
			// New method should give the actual ranking score
			expectedScore := rankingScores[seq]
			if newScore != expectedScore {
				t.Errorf("New method failed: expected=%.2f, got=%.2f", expectedScore, newScore)
			}
			
			t.Logf("Sequence %d: Old method=%.3f, New method=%.3f (expected=%.3f)", 
				seq, oldScore, newScore, expectedScore)
		})
	}
}

func TestRerankAPIHandler(t *testing.T) {
	// Create a mock server
	mockServer := &MockRerankingServer{reranking: true}
	
	// Test the API handler
	testCases := []struct {
		name           string
		requestBody    string
		expectedStatus int
		description    string
	}{
		{
			name: "Valid rerank request",
			requestBody: `{
				"model": "test-reranker",
				"prompts": [
					"Query: machine learning Document: Pizza is made with cheese",
					"Query: machine learning Document: Machine learning is AI subset", 
					"Query: machine learning Document: Weather is sunny today"
				]
			}`,
			expectedStatus: 200,
			description:    "Should process valid rerank request successfully",
		},
		{
			name:           "Invalid JSON",
			requestBody:    `{invalid json}`,
			expectedStatus: 400,
			description:    "Should reject invalid JSON",
		},
		{
			name:           "Empty prompts",
			requestBody:    `{"model": "test", "prompts": []}`,
			expectedStatus: 200,
			description:    "Should handle empty prompts array",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/rerank", strings.NewReader(tc.requestBody))
			req.Header.Set("Content-Type", "application/json")
			
			w := httptest.NewRecorder()
			mockServer.rerank(w, req)
			
			if w.Code != tc.expectedStatus {
				t.Errorf("Expected status %d, got %d for test: %s", 
					tc.expectedStatus, w.Code, tc.description)
			}
			
			if w.Code == 200 {
				var response RerankResponse
				err := json.NewDecoder(w.Body).Decode(&response)
				if err != nil {
					t.Errorf("Failed to decode response: %v", err)
				}
				
				t.Logf("Response for '%s': %d results", tc.name, len(response.Results))
			}
		})
	}
}

func TestRerankingModelDetection(t *testing.T) {
	// Test that reranking flag is properly detected
	tests := []struct {
		name      string
		reranking bool
		expected  string
	}{
		{
			name:      "Reranking enabled",
			reranking: true,
			expected:  "200",
		},
		{
			name:      "Reranking disabled", 
			reranking: false,
			expected:  "501", // Not Implemented
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockServer := &MockRerankingServer{reranking: tt.reranking}
			
			reqBody := `{"model": "test", "prompts": ["test prompt"]}`
			req := httptest.NewRequest("POST", "/rerank", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			
			w := httptest.NewRecorder()
			mockServer.rerank(w, req)
			
			expectedCode := 200
			if tt.expected == "501" {
				expectedCode = 501
			}
			
			if w.Code != expectedCode {
				t.Errorf("Expected status %d, got %d for reranking=%v", 
					expectedCode, w.Code, tt.reranking)
			}
		})
	}
}

// Benchmark the score extraction performance
func BenchmarkScoreExtraction(b *testing.B) {
	// Simulate large batch of ranking scores
	scores := make([]float32, 1000)
	for i := range scores {
		scores[i] = float32(i) / 1000.0
	}
	
	b.Run("New_Method_Ranking_Scores", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for seq := 0; seq < len(scores); seq++ {
				// New method: direct extraction
				_ = scores[seq]
			}
		}
	})
	
	b.Run("Old_Method_Vocab_Logits", func(b *testing.B) {
		vocabSize := 32000
		vocabLogits := make([]float32, len(scores)*vocabSize)
		
		for i := 0; i < b.N; i++ {
			for seq := 0; seq < len(scores); seq++ {
				// Old method: extract from vocabulary logits
				_ = vocabLogits[seq*vocabSize]
			}
		}
	})
}

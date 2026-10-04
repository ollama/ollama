package server

import (
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// BenchmarkOpenAIEmbeddingsCompatibility measures OpenAI request parsing,
// embedding response conversion, and JSON serialization with synthetic matrices.
// It excludes model inference and cloud middleware.
func BenchmarkOpenAIEmbeddingsCompatibility(b *testing.B) {
	gin.SetMode(gin.TestMode)
	for _, size := range [][2]int{{1, 384}, {32, 768}, {128, 1536}, {128, 4096}} {
		b.Run(fmt.Sprintf("%dx%d", size[0], size[1]), func(b *testing.B) {
			embeddings := make([][]float32, size[0])
			for i := range embeddings {
				embeddings[i] = make([]float32, size[1])
				for j := range embeddings[i] {
					embeddings[i][j] = float32((i+j)%101) / 101
				}
			}
			resp := api.EmbedResponse{Model: "benchmark-model", Embeddings: embeddings}
			router := gin.New()
			router.POST("/v1/embeddings", func(c *gin.Context) {
				var req openai.EmbedRequest
				if err := c.ShouldBindJSON(&req); err != nil {
					b.Fatal(err)
				}
				c.JSON(200, openai.ToEmbeddingList(req.Model, resp, req.EncodingFormat))
			})
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"model":"benchmark-model","input":"test","encoding_format":"base64"}`))
				w := httptest.NewRecorder()
				router.ServeHTTP(w, req)
				if w.Code != 200 {
					b.Fatal(w.Body.String())
				}
			}
		})
	}
}

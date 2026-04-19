package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cohere"
)

func TestCohereEmbedMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.EmbedRequest
		err  cohere.ErrorResponse
	}

	var capturedRequest *api.EmbedRequest

	testCases := []testCase{
		{
			name: "texts request",
			body: `{"model":"test-model","texts":["hello","world"]}`,
			req: api.EmbedRequest{
				Model: "test-model",
				Input: []any{"hello", "world"},
				Truncate: func() *bool {
					v := true
					return &v
				}(),
			},
		},
		{
			name: "images request",
			body: `{"model":"test-model","images":["` + prefix + image + `"],"input_type":"image"}`,
			req: api.EmbedRequest{
				Model:  "test-model",
				Inputs: []api.EmbedInput{{Image: mustDecodeBase64(t, image)}},
				Truncate: func() *bool {
					v := true
					return &v
				}(),
			},
		},
		{
			name: "mixed inputs request",
			body: `{"model":"test-model","inputs":[{"content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"` + prefix + image + `"}}]}],"embedding_types":["float","base64"],"output_dimension":512,"truncate":"NONE"}`,
			req: api.EmbedRequest{
				Model:      "test-model",
				Dimensions: 512,
				Inputs: []api.EmbedInput{{
					Text:  "hello",
					Image: mustDecodeBase64(t, image),
				}},
				Truncate: func() *bool {
					v := false
					return &v
				}(),
			},
		},
		{
			name: "unsupported embedding type",
			body: `{"model":"test-model","texts":["hello"],"embedding_types":["int8"]}`,
			err:  cohere.NewError(`embedding type "int8" is not supported`),
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(CohereEmbedMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/v2/embed", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v2/embed", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp cohere.ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			if capturedRequest != nil && !reflect.DeepEqual(tc.req, *capturedRequest) {
				t.Fatalf("requests did not match: got %#v want %#v", *capturedRequest, tc.req)
			}

			if !reflect.DeepEqual(tc.err, errResp) {
				t.Fatalf("errors did not match: got %#v want %#v", errResp, tc.err)
			}

			capturedRequest = nil
		})
	}
}

func TestCohereEmbedWriter(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)

	writer := &CohereEmbedWriter{
		BaseWriter: BaseWriter{ResponseWriter: context.Writer},
		req: cohere.EmbedRequest{
			Model:          "test-model",
			Images:         []string{prefix + image},
			EmbeddingTypes: []string{"float", "base64"},
		},
	}

	context.Writer.WriteHeader(http.StatusOK)
	payload, err := json.Marshal(api.EmbedResponse{
		Model:           "test-model",
		Embeddings:      [][]float32{{0.1, 0.2}},
		PromptEvalCount: 3,
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := writer.Write(payload); err != nil {
		t.Fatal(err)
	}

	var resp cohere.EmbedResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if got, want := len(resp.Embeddings.Float), 1; got != want {
		t.Fatalf("len(float embeddings) = %d, want %d", got, want)
	}
	if got, want := len(resp.Embeddings.Base64), 1; got != want {
		t.Fatalf("len(base64 embeddings) = %d, want %d", got, want)
	}
	if got, want := len(resp.Images), 1; got != want {
		t.Fatalf("len(images) = %d, want %d", got, want)
	}
	if resp.ResponseType != "embeddings_by_type" {
		t.Fatalf("response_type = %q, want embeddings_by_type", resp.ResponseType)
	}
}

func mustDecodeBase64(t *testing.T, value string) []byte {
	t.Helper()

	data, err := cohere.FromEmbedRequest(cohere.EmbedRequest{
		Model:  "test-model",
		Images: []string{prefix + value},
	})
	if err != nil {
		t.Fatal(err)
	}

	return data.Inputs[0].Image
}

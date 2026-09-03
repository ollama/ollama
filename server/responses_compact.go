package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
)

// responsesCompactionMiddleware intercepts only Codex compaction control items.
// Ordinary Responses requests continue through the existing route unchanged.
func (s *Server) responsesCompactionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := readResponsesCompactionBody(c)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		plan, requested, err := openai.PrepareTriggeredCompaction(body)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if requested {
			c.Abort()
			s.handleResponsesCompaction(c, plan, true)
			return
		}

		rewritten, changed, err := openai.ExpandResponsesCompactionInput(body)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if changed {
			resetResponsesRequestBody(c.Request, rewritten)
		}
		c.Next()
	}
}

// ResponsesCompactHandler implements POST /v1/responses/compact with an
// Ollama-owned ordinary inference request rather than upstream passthrough.
func (s *Server) ResponsesCompactHandler(c *gin.Context) {
	body, err := readResponsesCompactionBody(c)
	if err != nil {
		writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	plan, err := openai.PrepareStandaloneCompaction(body)
	if err != nil {
		writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.handleResponsesCompaction(c, plan, false)
}

func readResponsesCompactionBody(c *gin.Context) ([]byte, error) {
	if c.GetHeader("Content-Encoding") == "zstd" {
		reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
		if err != nil {
			return nil, fmt.Errorf("failed to decompress zstd body")
		}
		decompressed, err := io.ReadAll(http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize))
		reader.Close()
		if err != nil {
			return nil, err
		}
		c.Request.Header.Del("Content-Encoding")
		resetResponsesRequestBody(c.Request, decompressed)
	}

	body, err := readRequestBody(c.Request)
	if err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return nil, fmt.Errorf("missing request body")
	}
	return body, nil
}

func resetResponsesRequestBody(r *http.Request, body []byte) {
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
}

func (s *Server) handleResponsesCompaction(c *gin.Context, plan *openai.ResponsesCompactionPlan, stream bool) {
	var validationErr error
	for attempt := 0; attempt < 2; attempt++ {
		repair := ""
		if validationErr != nil {
			repair = validationErr.Error()
		}
		request, err := plan.SummaryRequest(repair)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusInternalServerError, "compaction_failed", "compaction failed; the original conversation is unchanged")
			return
		}

		response := s.runResponsesCompactionInference(c, request)
		if response.status < http.StatusOK || response.status >= http.StatusMultipleChoices {
			copyResponsesCompactionResponse(c, response)
			return
		}

		result, err := plan.Complete(response.body.Bytes())
		if err != nil {
			validationErr = err
			continue
		}

		id := fmt.Sprintf("resp_compact_%d", time.Now().UnixNano())
		if stream {
			writeResponsesCompactionStream(c, openai.NewResponsesCompactionStreamEvents(id, plan.Model, result))
			return
		}
		c.JSON(http.StatusOK, openai.NewResponsesCompactedResponse(id, result))
		return
	}

	writeResponsesCompactionError(c, http.StatusInternalServerError, "compaction_failed", "compaction failed; the selected model did not return a valid summary and the original conversation is unchanged")
}

// runResponsesCompactionInference uses the normal Responses stack without the
// compaction dispatcher. This keeps local and cloud model selection identical
// to an ordinary request and permits one isolated repair retry.
func (s *Server) runResponsesCompactionInference(c *gin.Context, body []byte) *responsesInferenceRecorder {
	router := gin.New()
	router.POST("/v1/responses",
		cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable),
		middleware.ResponsesMiddleware(),
		s.ChatHandler,
	)

	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, "/v1/responses", bytes.NewReader(body))
	if err != nil {
		return &responsesInferenceRecorder{header: make(http.Header), status: http.StatusInternalServerError, body: *bytes.NewBufferString(err.Error())}
	}
	req.Header = c.Request.Header.Clone()
	req.Header.Del("Content-Encoding")
	req.Header.Set("Content-Type", "application/json")
	req.ContentLength = int64(len(body))

	recorder := &responsesInferenceRecorder{header: make(http.Header)}
	router.ServeHTTP(recorder, req)
	return recorder
}

type responsesInferenceRecorder struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (r *responsesInferenceRecorder) Header() http.Header {
	return r.header
}

func (r *responsesInferenceRecorder) WriteHeader(status int) {
	if r.status == 0 {
		r.status = status
	}
}

func (r *responsesInferenceRecorder) Write(data []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	return r.body.Write(data)
}

func (r *responsesInferenceRecorder) Flush() {}

func copyResponsesCompactionResponse(c *gin.Context, response *responsesInferenceRecorder) {
	for key, values := range response.header {
		if key == "Content-Length" {
			continue
		}
		for _, value := range values {
			c.Header(key, value)
		}
	}
	c.Data(response.status, response.header.Get("Content-Type"), response.body.Bytes())
}

func writeResponsesCompactionStream(c *gin.Context, events []openai.ResponsesStreamEvent) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(http.StatusOK)
	for _, event := range events {
		data, err := json.Marshal(event.Data)
		if err != nil {
			return
		}
		_, _ = fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", event.Event, data)
	}
	if flusher, ok := c.Writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func writeResponsesCompactionError(c *gin.Context, status int, code, message string) {
	response := openai.NewError(status, message)
	response.Error.Code = &code
	c.AbortWithStatusJSON(status, response)
}

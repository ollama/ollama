package server

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) OpenAIEmbeddingsHandler(c *gin.Context) {
	var req openai.EmbedRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeOpenAIEmbedError(c, &embedError{http.StatusBadRequest, err.Error()})
		return
	}
	if req.EncodingFormat != "" && !strings.EqualFold(req.EncodingFormat, "float") && !strings.EqualFold(req.EncodingFormat, "base64") {
		writeOpenAIEmbedError(c, &embedError{http.StatusBadRequest, fmt.Sprintf("Invalid value for 'encoding_format' = %s. Supported values: ['float', 'base64'].", req.EncodingFormat)})
		return
	}
	// The old request round trip decoded []string{""} as []any{""}.
	// Keep the OpenAI empty-string semantics outside native input parsing.
	if req.Input == "" {
		req.Input = []any{""}
	}
	if req.Input == nil {
		writeOpenAIEmbedError(c, &embedError{http.StatusBadRequest, "invalid input"})
		return
	}
	if input, ok := req.Input.([]any); ok && len(input) == 0 {
		writeOpenAIEmbedError(c, &embedError{http.StatusBadRequest, "invalid input"})
		return
	}

	nativeReq := api.EmbedRequest{Model: req.Model, Input: req.Input, Dimensions: req.Dimensions}
	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		status, message := http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model)
		switch {
		case errors.Is(err, errConflictingModelSource):
			status, message = http.StatusBadRequest, err.Error()
		case errors.Is(err, model.ErrUnqualifiedName):
			status, message = http.StatusBadRequest, errtypes.InvalidModelNameErrMsg
		}
		writeOpenAIEmbedError(c, &embedError{status, message})
		return
	}
	// Normally intercepted by cloudPassthroughMiddleware; keep standalone
	// handler calls from ever dispatching a cloud reference to the local runner.
	if modelRef.Source == modelSourceCloud {
		req.Model = modelRef.Base
		proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
		return
	}
	input, err := parseEmbedInput(nativeReq.Input)
	if err != nil {
		writeOpenAIEmbedError(c, &embedError{http.StatusBadRequest, err.Error()})
		return
	}
	resp, err := s.embedLocal(c.Request.Context(), nativeReq, modelRef.Name, input)
	if err != nil {
		writeOpenAIEmbedError(c, err)
		return
	}
	c.JSON(http.StatusOK, openai.ToEmbeddingList(req.Model, resp, req.EncodingFormat))
}

func writeOpenAIEmbedError(c *gin.Context, err error) {
	status := http.StatusInternalServerError
	var e *embedError
	if errors.As(err, &e) {
		status = e.status
	}
	// Preserve BaseWriter's fallback message for an empty native error.
	message := (api.StatusError{ErrorMessage: err.Error()}).Error()
	c.AbortWithStatusJSON(status, openai.NewError(status, message))
}

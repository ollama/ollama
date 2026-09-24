package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type entityExtractor interface {
	Extract(context.Context, api.ExtractRequest) (*api.ExtractResponse, error)
}

func (s *Server) ExtractHandler(c *gin.Context) {
	start := time.Now()
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 2<<20)
	var req api.ExtractRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := req.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	ref, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if ref.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "entity extraction requires a local model"})
		return
	}
	name, err := getExistingName(ref.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}
	m, err := GetModel(name.String())
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	r, _, _, err := s.scheduleRunner(c.Request.Context(), m, []model.Capability{model.CapabilityExtraction}, nil, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	loaded := time.Now()
	extractor, ok := r.(entityExtractor)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model runner does not support extraction"})
		return
	}
	resp, err := extractor.Extract(c.Request.Context(), req)
	if err != nil {
		var status api.StatusError
		if errors.As(err, &status) {
			c.JSON(status.StatusCode, gin.H{"error": status.ErrorMessage})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
	resp.Model = req.Model
	resp.TotalDuration = time.Since(start)
	resp.LoadDuration = loaded.Sub(start)
	c.JSON(http.StatusOK, resp)
}

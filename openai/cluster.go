// Package openai provides middleware for partial compatibility with the OpenAI REST API
// This file extends the compatibility layer to include cluster operations
package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

// ClusterModelRequest represents the OpenAI-compatible request format for cluster model loading
type ClusterModelRequest struct {
	Model       string   `json:"model"`
	Distributed bool     `json:"distributed,omitempty"`
	ShardCount  int      `json:"shard_count,omitempty"`
	Strategy    string   `json:"strategy,omitempty"`
	NodeIDs     []string `json:"node_ids,omitempty"`
}

// ClusterModelResponse represents the OpenAI-compatible response for cluster model loading
type ClusterModelResponse struct {
	ID        string   `json:"id"`
	Object    string   `json:"object"`
	Created   int64    `json:"created"`
	Model     string   `json:"model"`
	Success   bool     `json:"success"`
	Distributed bool   `json:"distributed"`
	Nodes     []string `json:"nodes,omitempty"`
}

// ClusterStatusResponse represents the OpenAI-compatible response for cluster status
type ClusterStatusResponse struct {
	ID        string   `json:"id"`
	Object    string   `json:"object"`
	Created   int64    `json:"created"`
	Enabled   bool     `json:"enabled"`
	Mode      string   `json:"mode"`
	NodeCount int      `json:"node_count"`
	Healthy   bool     `json:"healthy"`
}

// BaseClusterWriter provides base functionality for cluster response writers
type BaseClusterWriter struct {
	BaseWriter
}

// ClusterModelWriter converts Ollama cluster model responses to OpenAI format
type ClusterModelWriter struct {
	id string
	BaseClusterWriter
}

// ClusterStatusWriter converts Ollama cluster status responses to OpenAI format
type ClusterStatusWriter struct {
	id string
	BaseClusterWriter
}

// ClusterNodeWriter converts Ollama cluster node responses to OpenAI format
type ClusterNodeWriter struct {
	id string
	BaseClusterWriter
}

// Write implements the io.Writer interface for ClusterModelWriter
func (w *ClusterModelWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	var clusterModelResponse api.ClusterModelLoadResponse
	err := json.Unmarshal(data, &clusterModelResponse)
	if err != nil {
		return 0, err
	}

	// Convert to OpenAI format
	openAIResponse := ClusterModelResponse{
		ID:        fmt.Sprintf("clustermodel-%d", rand.Intn(999)),
		Object:    "cluster.model",
		Created:   time.Now().Unix(),
		Model:     clusterModelResponse.Model,
		Success:   clusterModelResponse.Success,
		Distributed: clusterModelResponse.Distributed,
		Nodes:     clusterModelResponse.Nodes,
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openAIResponse)
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implements the io.Writer interface for ClusterStatusWriter
func (w *ClusterStatusWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	var statusResponse api.ClusterStatusResponse
	err := json.Unmarshal(data, &statusResponse)
	if err != nil {
		return 0, err
	}

	// Convert to OpenAI format
	openAIResponse := ClusterStatusResponse{
		ID:        fmt.Sprintf("clusterstatus-%d", rand.Intn(999)),
		Object:    "cluster.status",
		Created:   time.Now().Unix(),
		Enabled:   statusResponse.Enabled,
		Mode:      statusResponse.Mode,
		NodeCount: statusResponse.NodeCount,
		Healthy:   statusResponse.Healthy,
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openAIResponse)
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// Write implements the io.Writer interface for ClusterNodeWriter
func (w *ClusterNodeWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	// This endpoint returns an array of nodes directly
	var nodesResponse []api.ClusterNodeResponse
	err := json.Unmarshal(data, &nodesResponse)
	if err != nil {
		return 0, err
	}

	// Wrap the nodes in a list response for OpenAI format compatibility
	type NodeObject struct {
		ID          string `json:"id"`
		Name        string `json:"name"`
		Role        string `json:"role"`
		Status      string `json:"status"`
		Address     string `json:"address"`
		Models      []string `json:"models,omitempty"`
	}

	type NodesResponse struct {
		Object string       `json:"object"`
		Data   []NodeObject `json:"data"`
	}

	nodeObjects := make([]NodeObject, len(nodesResponse))
	for i, node := range nodesResponse {
		nodeObjects[i] = NodeObject{
			ID:      node.ID,
			Name:    node.Name,
			Role:    node.Role,
			Status:  node.Status,
			Address: node.Address,
			Models:  node.Models,
		}
	}

	openAIResponse := NodesResponse{
		Object: "list",
		Data:   nodeObjects,
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openAIResponse)
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// ClusterModelLoadMiddleware converts OpenAI API requests to Ollama cluster format
func ClusterModelLoadMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ClusterModelRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Convert to Ollama format
		ollamaReq := api.ClusterModelLoadRequest{
			Model:       req.Model,
			Distributed: req.Distributed,
			ShardCount:  req.ShardCount,
			Strategy:    req.Strategy,
			NodeIDs:     req.NodeIDs,
		}

		// Replace request body
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(ollamaReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		// Set up writer to convert response
		w := &ClusterModelWriter{
			id: fmt.Sprintf("clustermodel-%d", rand.Intn(999)),
			BaseClusterWriter: BaseClusterWriter{
				BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			},
		}

		c.Writer = w
		c.Next()
	}
}

// ClusterStatusMiddleware handles the OpenAI API compatibility for cluster status
func ClusterStatusMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// No request body to process for GET requests

		// Set up writer to convert response
		w := &ClusterStatusWriter{
			id: fmt.Sprintf("clusterstatus-%d", rand.Intn(999)),
			BaseClusterWriter: BaseClusterWriter{
				BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			},
		}

		c.Writer = w
		c.Next()
	}
}

// ClusterNodesMiddleware handles the OpenAI API compatibility for listing nodes
func ClusterNodesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// No request body to process for GET requests

		// Set up writer to convert response
		w := &ClusterNodeWriter{
			id: fmt.Sprintf("clusternodes-%d", rand.Intn(999)),
			BaseClusterWriter: BaseClusterWriter{
				BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			},
		}

		c.Writer = w
		c.Next()
	}
}

// ClusterGenerateMiddleware adapts the standard OpenAI completion format for cluster generation
func ClusterGenerateMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Read and process the request
		bodyBytes, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
			return
		}
		
		// Just pass through the body since the format is the same as standard generation
		// We're just routing it through the cluster endpoint instead
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		
		// The response will go through the standard openai.CompleteWriter or openai.ChatWriter
		c.Next()
	}
}
package server

import (
	"net/http"
	"sort"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/remoteproviders"
)

func (s *Server) ListRemoteProvidersHandler(c *gin.Context) {
	channels, err := remoteproviders.ListChannels(envconfig.RemoteProvidersPath())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	sort.Slice(channels, func(i, j int) bool {
		return channels[i].ID < channels[j].ID
	})

	resp := api.RemoteProviderListResponse{
		Providers: make([]api.RemoteProviderResponse, 0, len(channels)),
	}
	for _, ch := range channels {
		resp.Providers = append(resp.Providers, toRemoteProviderResponse(ch))
	}

	c.JSON(http.StatusOK, resp)
}

func (s *Server) UpsertRemoteProviderHandler(c *gin.Context) {
	var req api.RemoteProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	channel := remoteproviders.Channel{
		ID:           req.ID,
		Type:         req.Type,
		BaseURL:      req.BaseURL,
		APIKey:       req.APIKey,
		DefaultModel: req.DefaultModel,
		Headers:      req.Headers,
	}

	updated, err := remoteproviders.UpsertChannel(envconfig.RemoteProvidersPath(), channel)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, toRemoteProviderResponse(updated))
}

func (s *Server) DeleteRemoteProviderHandler(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "id is required"})
		return
	}

	if err := remoteproviders.DeleteChannel(envconfig.RemoteProvidersPath(), id); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.Status(http.StatusNoContent)
}

func toRemoteProviderResponse(ch remoteproviders.Channel) api.RemoteProviderResponse {
	hasKey := ch.APIKey != ""
	redacted := remoteproviders.RedactChannel(ch)

	return api.RemoteProviderResponse{
		ID:           redacted.ID,
		Type:         redacted.Type,
		BaseURL:      redacted.BaseURL,
		DefaultModel: redacted.DefaultModel,
		Headers:      redacted.Headers,
		APIKeyMasked: redacted.APIKey,
		HasAPIKey:    hasKey,
	}
}

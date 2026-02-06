package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/types/model"
)

type aliasListResponse struct {
	Aliases []aliasEntry `json:"aliases"`
}

type aliasDeleteRequest struct {
	Alias string `json:"alias"`
}

func (s *Server) ListAliasesHandler(c *gin.Context) {
	store, err := s.aliasStore()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var aliases []aliasEntry
	if store != nil {
		aliases = store.List()
	}

	c.JSON(http.StatusOK, aliasListResponse{Aliases: aliases})
}

func (s *Server) CreateAliasHandler(c *gin.Context) {
	var req aliasEntry
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	req.Alias = strings.TrimSpace(req.Alias)
	req.Target = strings.TrimSpace(req.Target)
	if req.Alias == "" || req.Target == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "alias and target are required"})
		return
	}

	// Target must always be a valid model name
	targetName := model.ParseName(req.Target)
	if !targetName.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("target %q is invalid", req.Target)})
		return
	}

	var aliasName model.Name
	if req.PrefixMatching {
		// For prefix aliases, we still parse the alias to normalize it,
		// but we allow any non-empty string since prefix patterns may not be valid model names
		aliasName = model.ParseName(req.Alias)
		// Even if not valid as a model name, we accept it for prefix matching
	} else {
		aliasName = model.ParseName(req.Alias)
		if !aliasName.IsValid() {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("alias %q is invalid", req.Alias)})
			return
		}

		if normalizeAliasKey(aliasName) == normalizeAliasKey(targetName) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "alias cannot point to itself"})
			return
		}

		exists, err := localModelExists(aliasName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if exists {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("alias %q conflicts with existing model", req.Alias)})
			return
		}
	}

	store, err := s.aliasStore()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if err := store.Set(aliasName, targetName, req.PrefixMatching); err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, errAliasCycle) {
			status = http.StatusBadRequest
		}
		c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
		return
	}

	resp := aliasEntry{
		Alias:          displayAliasName(aliasName),
		Target:         displayAliasName(targetName),
		PrefixMatching: req.PrefixMatching,
	}
	if req.PrefixMatching && !aliasName.IsValid() {
		// For prefix aliases that aren't valid model names, use the raw alias
		resp.Alias = req.Alias
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) DeleteAliasHandler(c *gin.Context) {
	var req aliasDeleteRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	req.Alias = strings.TrimSpace(req.Alias)
	if req.Alias == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "alias is required"})
		return
	}

	store, err := s.aliasStore()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	aliasName := model.ParseName(req.Alias)
	var deleted bool
	if aliasName.IsValid() {
		deleted, err = store.Delete(aliasName)
	} else {
		// For invalid model names (like prefix aliases), try deleting by raw string
		deleted, err = store.DeleteByString(req.Alias)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if !deleted {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("alias %q not found", req.Alias)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"deleted": true})
}

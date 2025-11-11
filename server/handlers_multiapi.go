package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api/context"
	"github.com/ollama/ollama/api/providers"
	"github.com/ollama/ollama/features"
	"github.com/ollama/ollama/rag"
	"github.com/ollama/ollama/templates"
	"github.com/ollama/ollama/workspace"
)

// Global agent sessions storage
var (
	agentSessions   = make(map[string]*agent.Session)
	agentSessionsMu sync.RWMutex
)

// Provider handlers (Phase 1)

type AddProviderRequest struct {
	Type    string `json:"type" binding:"required"`
	Name    string `json:"name" binding:"required"`
	APIKey  string `json:"api_key" binding:"required"`
	BaseURL string `json:"base_url,omitempty"`
}

func (s *Server) ListProvidersHandler(c *gin.Context) {
	// Return list of configured providers
	// TODO: Store providers in database and retrieve them
	c.JSON(http.StatusOK, gin.H{
		"providers": []gin.H{
			{"type": "openai", "name": "OpenAI"},
			{"type": "anthropic", "name": "Anthropic"},
			{"type": "google", "name": "Google Gemini"},
			{"type": "groq", "name": "Groq"},
		},
	})
}

func (s *Server) AddProviderHandler(c *gin.Context) {
	var req AddProviderRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	provider, err := providers.CreateProvider(req.Type, req.APIKey, req.BaseURL)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate credentials
	ctx := c.Request.Context()
	if err := provider.ValidateCredentials(ctx); err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid credentials: " + err.Error()})
		return
	}

	// TODO: Store provider in database
	c.JSON(http.StatusOK, gin.H{
		"message": "provider added successfully",
		"type":    req.Type,
		"name":    req.Name,
	})
}

func (s *Server) DeleteProviderHandler(c *gin.Context) {
	id := c.Param("id")
	// TODO: Remove provider from database
	c.JSON(http.StatusOK, gin.H{"message": "provider deleted", "id": id})
}

type ProviderChatRequest struct {
	Model       string                   `json:"model" binding:"required"`
	Messages    []providers.Message      `json:"messages" binding:"required"`
	Temperature *float64                 `json:"temperature,omitempty"`
	MaxTokens   *int                     `json:"max_tokens,omitempty"`
	APIKey      string                   `json:"api_key,omitempty"`
}

func (s *Server) ProviderChatHandler(c *gin.Context) {
	providerType := c.Param("provider")

	var req ProviderChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Create provider instance
	provider, err := providers.CreateProvider(providerType, req.APIKey, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Make chat request
	ctx := c.Request.Context()
	chatReq := providers.ChatRequest{
		Model:       req.Model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}

	response, err := provider.ChatCompletion(ctx, chatReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (s *Server) ProviderModelsHandler(c *gin.Context) {
	providerType := c.Param("provider")
	apiKey := c.Query("api_key")

	if apiKey == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "api_key is required"})
		return
	}

	provider, err := providers.CreateProvider(providerType, apiKey, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()
	models, err := provider.ListModels(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"models": models})
}

func (s *Server) ValidateProviderHandler(c *gin.Context) {
	providerType := c.Param("provider")

	var req struct {
		APIKey string `json:"api_key" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	provider, err := providers.CreateProvider(providerType, req.APIKey, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()
	if err := provider.ValidateCredentials(ctx); err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"valid": false, "error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"valid": true})
}

// Workspace handlers (Phase 2)

func (s *Server) GetWorkspaceRulesHandler(c *gin.Context) {
	workspacePath := c.Query("workspace")
	if workspacePath == "" {
		workspacePath = "." // default to current directory
	}

	manager := workspace.NewManager()
	rulesManager := workspace.NewRulesManager(manager)

	rules, err := rulesManager.GetRules(workspacePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"rules":         rules,
		"system_prompt": rules.ToSystemPrompt(),
	})
}

func (s *Server) UpdateWorkspaceRulesHandler(c *gin.Context) {
	workspacePath := c.Query("workspace")
	if workspacePath == "" {
		workspacePath = "."
	}

	var req struct {
		Content string `json:"content" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	manager := workspace.NewManager()
	rulesManager := workspace.NewRulesManager(manager)

	if err := rulesManager.UpdateRules(workspacePath, req.Content); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "rules updated successfully"})
}

func (s *Server) GetWorkspaceTodosHandler(c *gin.Context) {
	workspacePath := c.Query("workspace")
	if workspacePath == "" {
		workspacePath = "."
	}

	manager := workspace.NewManager()
	todoManager := workspace.NewTodoManager(manager)

	todos, err := todoManager.GetTodos(workspacePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, todos)
}

func (s *Server) CompleteWorkspaceTodoHandler(c *gin.Context) {
	workspacePath := c.Query("workspace")
	if workspacePath == "" {
		workspacePath = "."
	}

	var req struct {
		PhaseIndex int `json:"phase_index" binding:"required"`
		TaskIndex  int `json:"task_index" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	manager := workspace.NewManager()
	todoManager := workspace.NewTodoManager(manager)

	if err := todoManager.MarkTaskComplete(workspacePath, req.PhaseIndex, req.TaskIndex); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "task marked as complete"})
}

// RAG handlers (Phase 6)

func (s *Server) RAGIngestHandler(c *gin.Context) {
	var req struct {
		WorkspacePath string `json:"workspace_path" binding:"required"`
		Title         string `json:"title" binding:"required"`
		Content       string `json:"content" binding:"required"`
		Metadata      map[string]string `json:"metadata,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ragManager := rag.NewManager(req.WorkspacePath)
	ctx := c.Request.Context()

	doc := &rag.Document{
		Title:    req.Title,
		Content:  req.Content,
		Metadata: req.Metadata,
	}

	if err := ragManager.IngestDocument(ctx, doc); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "document ingested successfully"})
}

func (s *Server) RAGSearchHandler(c *gin.Context) {
	var req struct {
		WorkspacePath string `json:"workspace_path" binding:"required"`
		Query         string `json:"query" binding:"required"`
		TopK          int    `json:"top_k,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopK == 0 {
		req.TopK = 5
	}

	ragManager := rag.NewManager(req.WorkspacePath)
	ctx := c.Request.Context()

	results, err := ragManager.Search(ctx, req.Query, req.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"results": results})
}

// Template handlers (Phase 5)

func (s *Server) ListTemplatesHandler(c *gin.Context) {
	manager := templates.NewManager()
	templateList := manager.List()

	c.JSON(http.StatusOK, gin.H{"templates": templateList})
}

func (s *Server) RenderTemplateHandler(c *gin.Context) {
	var req struct {
		TemplateID string            `json:"template_id" binding:"required"`
		Variables  map[string]string `json:"variables" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	manager := templates.NewManager()
	rendered, err := manager.Render(req.TemplateID, req.Variables)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"rendered": rendered})
}

// Agent handlers (Phase 10)

func (s *Server) StartAgentSessionHandler(c *gin.Context) {
	var req struct {
		WorkspacePath    string `json:"workspace_path" binding:"required"`
		SupervisorAPIKey string `json:"supervisor_api_key" binding:"required"`
		WorkerAPIKey     string `json:"worker_api_key" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	supervisorProvider, err := providers.CreateProvider("anthropic", req.SupervisorAPIKey, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "supervisor provider error: " + err.Error()})
		return
	}

	workerProvider, err := providers.CreateProvider("anthropic", req.WorkerAPIKey, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "worker provider error: " + err.Error()})
		return
	}

	// Create workspace managers
	workspaceManager := workspace.NewManager()
	todoManager := workspace.NewTodoManager(workspaceManager)
	rulesManager := workspace.NewRulesManager(workspaceManager)

	controller := agent.NewController(supervisorProvider, workerProvider, todoManager, rulesManager)
	ctx := c.Request.Context()

	session, err := controller.StartSession(ctx, req.WorkspacePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Store session for later retrieval
	agentSessionsMu.Lock()
	agentSessions[session.ID] = session
	agentSessionsMu.Unlock()

	c.JSON(http.StatusOK, session)
}

func (s *Server) GetAgentStatusHandler(c *gin.Context) {
	sessionID := c.Param("id")

	agentSessionsMu.RLock()
	session, exists := agentSessions[sessionID]
	agentSessionsMu.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
		return
	}

	c.JSON(http.StatusOK, session)
}

// Voice I/O handlers (Phase 11)

func (s *Server) VoiceTranscribeHandler(c *gin.Context) {
	apiKey := c.PostForm("api_key")
	if apiKey == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "api_key is required"})
		return
	}

	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file is required"})
		return
	}
	defer file.Close()

	voiceHandler := features.NewVoiceHandler(apiKey)
	text, err := voiceHandler.Transcribe(file, header.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"text": text})
}

func (s *Server) VoiceSynthesizeHandler(c *gin.Context) {
	var req struct {
		APIKey string `json:"api_key" binding:"required"`
		Text   string `json:"text" binding:"required"`
		Voice  string `json:"voice,omitempty"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	voiceHandler := features.NewVoiceHandler(req.APIKey)
	audioData, err := voiceHandler.Synthesize(req.Text, req.Voice)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Type", "audio/mpeg")
	c.Header("Content-Disposition", "attachment; filename=speech.mp3")
	c.Data(http.StatusOK, "audio/mpeg", audioData)
}

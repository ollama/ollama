package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

// ToolsHandler handles requests to list available MCP tools.
// GET: Returns available MCP server definitions from configuration.
// POST with mcp_servers: Returns tools from the specified MCP servers.
func (s *Server) ToolsHandler(c *gin.Context) {
	var req struct {
		MCPServers []api.MCPServerConfig `json:"mcp_servers,omitempty"`
	}

	if c.Request.Method == "POST" {
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	
	// If MCP servers provided, list their tools
	if len(req.MCPServers) > 0 {
		manager := NewMCPManager(10)
		defer manager.Close()
		
		var allTools []ToolInfo
		for _, config := range req.MCPServers {
			if err := manager.AddServer(config); err != nil {
				// Include error in response but continue
				allTools = append(allTools, ToolInfo{
					Name:        config.Name,
					Description: "Failed to initialize: " + err.Error(),
					Error:       err.Error(),
				})
				continue
			}
			
			// Get tools from this server
			tools := manager.GetAllTools()
			for _, tool := range tools {
				allTools = append(allTools, ToolInfo{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  &tool.Function.Parameters,
					ServerName:  config.Name,
				})
			}
		}
		
		c.JSON(http.StatusOK, ToolsResponse{
			Tools: allTools,
		})
		return
	}
	
	// Otherwise, list available MCP server definitions
	defs, err := LoadMCPDefinitions()
	if err != nil {
		// Config parsing errors are client errors (bad config), not server errors
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid MCP configuration: " + err.Error()})
		return
	}

	servers := defs.ListServers()
	c.JSON(http.StatusOK, MCPServersResponse{
		Servers: servers,
	})
}

// ToolInfo provides information about a single tool
type ToolInfo struct {
	Name        string                      `json:"name"`
	Description string                      `json:"description"`
	Parameters  *api.ToolFunctionParameters `json:"parameters,omitempty"`
	ServerName  string                      `json:"server,omitempty"`
	Error       string                      `json:"error,omitempty"`
}

// ToolsResponse contains the list of available tools
type ToolsResponse struct {
	Tools []ToolInfo `json:"tools"`
}

// MCPServersResponse contains the list of available MCP server types
type MCPServersResponse struct {
	Servers []MCPServerInfo `json:"servers"`
}
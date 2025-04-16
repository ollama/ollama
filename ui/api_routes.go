package ui

import (
	"encoding/json"
	"net/http"
)

// APIRoutes registers the API routes with the HTTP server
func APIRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/models", handleModels)
	mux.HandleFunc("/api/download/", handleDownload)
	mux.HandleFunc("/api/status/", handleStatus)
	mux.HandleFunc("/api/remove/", handleRemove)
	mux.HandleFunc("/api/cancel/", handleCancel)
}

// Handlers for API endpoints
func handleModels(w http.ResponseWriter, r *http.Request) {
	// Implementation in minimal_server.go for now
}

func handleDownload(w http.ResponseWriter, r *http.Request) {
	// Implementation in minimal_server.go for now
}

func handleStatus(w http.ResponseWriter, r *http.Request) {
	// Implementation in minimal_server.go for now
}

func handleRemove(w http.ResponseWriter, r *http.Request) {
	// Implementation in minimal_server.go for now
}

func handleCancel(w http.ResponseWriter, r *http.Request) {
	// Implementation in minimal_server.go for now
}

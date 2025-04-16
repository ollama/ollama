package ui

import (
	"net/http"
)

// Dashboard registers the web dashboard routes
func Dashboard(mux *http.ServeMux) {
	mux.HandleFunc("/", handleDashboard)
}

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	// Implementation to come in future versions
	// For now, redirect to API documentation
	http.Redirect(w, r, "/docs", http.StatusFound)
}

//go:build windows || darwin

package ui

import (
	"bytes"
	"embed"
	"errors"
	"io/fs"
	"net/http"
	"strings"
	"time"
)

//go:embed app/dist
var appFS embed.FS

// appHandler returns an HTTP handler that serves the React SPA.
// It tries to serve real files first, then falls back to index.html for React Router.
func (s *Server) appHandler() http.Handler {
	// Strip the dist prefix so URLs look clean
	fsys, _ := fs.Sub(appFS, "app/dist")
	fileServer := http.FileServer(http.FS(fsys))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(r.URL.Path, "/")
		if _, err := fsys.Open(p); err == nil {
			// Serve the file directly
			fileServer.ServeHTTP(w, r)
			return
		}
		// Fallback – serve index.html for unknown paths so React Router works
		data, err := fs.ReadFile(fsys, "index.html")
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				http.NotFound(w, r)
			} else {
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
			return
		}
		http.ServeContent(w, r, "index.html", time.Time{}, bytes.NewReader(data))
	})
}

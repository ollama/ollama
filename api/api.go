package api

import (
	"errors"
	"fmt"
	"net/http"
	"os"

	"bllamo.com/build"
	"bllamo.com/build/blob"
	"bllamo.com/client/ollama/apitype"
	"bllamo.com/oweb"
	"bllamo.com/registry"
)

// Common API Errors
var (
	errUnqualifiedRef = oweb.Mistake("invalid", "name", "must be fully qualified")
	errRefNotFound    = oweb.Mistake("not_found", "name", "no such model")
)

type Server struct {
	Build *build.Server
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	oweb.Serve(s.serveHTTP, w, r)
}

func (s *Server) serveHTTP(w http.ResponseWriter, r *http.Request) error {
	switch r.URL.Path {
	case "/v1/push":
		return s.handlePush(w, r)
	default:
		return oweb.ErrNotFound
	}
}

func want(r *http.Request, method, path string) bool {
	return r.Method == method && r.URL.Path == path
}

func (s *Server) handlePush(_ http.ResponseWriter, r *http.Request) error {
	if r.Method != "POST" {
		return oweb.ErrMethodNotAllowed
	}

	params, err := oweb.DecodeJSON[apitype.PushRequest](r.Body)
	if err != nil {
		return err
	}

	if params.Name == "" {
		return oweb.Missing("name")
	}

	const registryURLTODO = "http://localhost:8888"

	ref := blob.ParseRef(params.Name)
	if !ref.FullyQualified() {
		return errUnqualifiedRef
	}

	man, err := s.Build.Manifest(ref)
	if err != nil {
		if errors.Is(err, build.ErrNotFound) {
			return errRefNotFound
		}
		return err
	}

	c := registry.Client{BaseURL: registryURLTODO}
	requirements, err := c.Push(r.Context(), params.Name, man)
	if err != nil {
		return err
	}

	for _, rq := range requirements {
		l, err := s.Build.LayerFile(rq.Digest)
		if err != nil {
			return err
		}
		err = func() error {
			f, err := os.Open(l)
			if err != nil {
				return err
			}
			defer f.Close()
			return registry.PushLayer(r.Context(), rq.URL, rq.Size, f)
		}()
		if err != nil {
			return err
		}
	}

	// commit the manifest to the registry
	requirements, err = c.Push(r.Context(), params.Name, man)
	if err != nil {
		return err
	}
	for _, r := range requirements {
		err = errors.Join(err, fmt.Errorf("push failed for %q", r.Digest))
	}
	return err

}

package api

import (
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/ollama/ollama/x/build"
	"github.com/ollama/ollama/x/client/ollama/apitype"
	"github.com/ollama/ollama/x/oweb"
	"github.com/ollama/ollama/x/registry"
	regtype "github.com/ollama/ollama/x/registry/apitype"
)

// Common API Errors
var (
	errUnqualifiedRef = oweb.Invalid("invalid", "name", "must be fully qualified")
	errRefNotFound    = oweb.Invalid("not_found", "name", "no such model")
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

	man, err := s.Build.ManifestData(params.Name)
	if err != nil {
		if errors.Is(err, build.ErrNotFound) {
			return errRefNotFound
		}
		return err
	}

	c := registry.Client{BaseURL: registryURLTODO}
	requirements, err := c.Push(r.Context(), params.Name, man, nil)
	if err != nil {
		return err
	}

	var uploads []regtype.CompletePart
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
			cp, err := registry.PushLayer(r.Context(), f, rq.URL, rq.Offset, rq.Size)
			if err != nil {
				return err
			}
			uploads = append(uploads, cp)
			return nil
		}()
		if err != nil {
			return err
		}
	}

	// commit the manifest to the registry
	requirements, err = c.Push(r.Context(), params.Name, man, &registry.PushParams{
		CompleteParts: uploads,
	})
	if err != nil {
		return err
	}
	for _, r := range requirements {
		err = errors.Join(err, fmt.Errorf("push failed for %q", r.Digest))
	}
	return err

}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) error {
	return oweb.ErrNotFound
}

// Package implements an Ollama registry client and server package registry
package registry

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"path"
	"time"

	"bllamo.com/build/blob"
	"bllamo.com/client/ollama"
	"bllamo.com/oweb"
	"bllamo.com/registry/apitype"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// TODO(bmizerany): move all env things to package envkobs?
var defaultLibrary = cmp.Or(os.Getenv("OLLAMA_REGISTRY"), "registry.ollama.ai/library")

func DefaultLibrary() string {
	return defaultLibrary
}

type Server struct {
	minioClient *minio.Client
}

func New(mc *minio.Client) *Server {
	return &Server{minioClient: mc}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := s.serveHTTP(w, r); err != nil {
		log.Printf("error: %v", err) // TODO(bmizerany): take a slog.Logger
		var e *ollama.Error
		if !errors.As(err, &e) {
			e = oweb.ErrInternal
		}
		w.WriteHeader(cmp.Or(e.Status, 400))
		if err := oweb.EncodeJSON(w, e); err != nil {
			log.Printf("error encoding error: %v", err)
		}
	}
}

func (s *Server) serveHTTP(w http.ResponseWriter, r *http.Request) error {
	switch r.URL.Path {
	case "/v1/push":
		return s.handlePush(w, r)
	case "/v1/pull":
		return s.handlePull(w, r)
	default:
		return oweb.ErrNotFound
	}
}

func (s *Server) handlePush(w http.ResponseWriter, r *http.Request) error {
	pr, err := oweb.DecodeUserJSON[apitype.PushRequest]("", r.Body)
	if err != nil {
		return err
	}

	ref := blob.ParseRef(pr.Ref)
	if !ref.Complete() {
		return oweb.Mistake("invalid", "name", "must be fully qualified")
	}

	m, err := oweb.DecodeUserJSON[apitype.Manifest]("manifest", bytes.NewReader(pr.Manifest))
	if err != nil {
		return err
	}

	// TODO(bmizerany): parallelize
	const chunkSizeTODO = 50 * 1024 * 1024
	var requirements []apitype.Requirement
	for _, l := range m.Layers {
		if l.Size == 0 {
			continue
		}

		// TODO(bmizerany): "global" throttle of rate of transfer

		pushed, err := s.statObject(r.Context(), l.Digest)
		if err != nil {
			return err
		}
		if !pushed {
			const expires = 15 * time.Minute
			key := path.Join("blobs", l.Digest)
			signedURL, err := s.mc().PresignedPutObject(r.Context(), "test", key, expires)
			if err != nil {
				return err
			}

			size := min(l.Size, chunkSizeTODO)
			requirements = append(requirements, apitype.Requirement{
				Digest: l.Digest,
				Size:   size,

				// TODO(bmizerany): use signed+temp urls
				URL: signedURL.String(),
			})
		}
	}

	if len(requirements) == 0 {
		// Commit the manifest
		body := bytes.NewReader(pr.Manifest)
		path := path.Join("manifests", path.Join(ref.Parts()...))
		_, err := s.mc().PutObject(r.Context(), "test", path, body, int64(len(pr.Manifest)), minio.PutObjectOptions{})
		if err != nil {
			return err
		}
	}

	return oweb.EncodeJSON(w, &apitype.PushResponse{Requirements: requirements})
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) error {
	// lookup manifest
	panic("TODO")
}

func (s *Server) statObject(ctx context.Context, digest string) (pushed bool, err error) {
	// HEAD the object
	path := path.Join("blobs", digest)
	_, err = s.mc().StatObject(ctx, "test", path, minio.StatObjectOptions{})
	if err != nil {
		if isNoSuchKey(err) {
			err = nil
		}
		return false, err
	}
	return true, nil
}

func isNoSuchKey(err error) bool {
	var e minio.ErrorResponse
	return errors.As(err, &e) && e.Code == "NoSuchKey"
}

func (s *Server) mc() *minio.Client {
	if s.minioClient != nil {
		return s.minioClient
	}
	mc, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		panic(err)
	}
	return mc
}

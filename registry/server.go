// Package implements an Ollama registry client and server
package registry

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"log"
	"net/http"
	"time"

	"bllamo.com/build"
	"bllamo.com/build/blob"
	"bllamo.com/client/ollama"
	"bllamo.com/oweb"
	"bllamo.com/registry/apitype"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type Server struct{}

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
	if !ref.FullyQualified() {
		return oweb.Mistake("invalid", "name", "must be fully qualified")
	}

	mc, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})

	m, err := oweb.DecodeUserJSON[apitype.Manifest]("manifest", bytes.NewReader(pr.Manifest))
	if err != nil {
		return err
	}

	// TODO(bmizerany): parallelize
	var requirements []apitype.Requirement
	for _, l := range m.Layers {
		if l.Size == 0 {
			continue
		}

		pushed, err := s.statObject(r.Context(), l.Digest)
		if err != nil {
			return err
		}
		if !pushed {
			const expires = 1 * time.Hour
			signedURL, err := mc.PresignedPutObject(r.Context(), "test", l.Digest, expires)
			if err != nil {
				return err
			}
			requirements = append(requirements, apitype.Requirement{
				Digest: l.Digest,
				Size:   l.Size,

				// TODO(bmizerany): use signed+temp urls
				URL: signedURL.String(),
			})
		}
	}

	if len(requirements) == 0 {
		const cheatTODO = "registry.ollama.ai/library"
		key := build.ManifestKey(cheatTODO, ref)
		_, err := mc.PutObject(r.Context(), "test", key, bytes.NewReader(pr.Manifest), int64(len(pr.Manifest)), minio.PutObjectOptions{})
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
	// TODO(bmizerany): hold client on *Server (hack for now)
	mc, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		return false, err
	}

	// HEAD the object
	_, err = mc.StatObject(ctx, "test", digest, minio.StatObjectOptions{})
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

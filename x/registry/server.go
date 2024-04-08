// Package implements an Ollama registry client and server package registry
package registry

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/ollama/ollama/x/client/ollama"
	"github.com/ollama/ollama/x/model"
	"github.com/ollama/ollama/x/oweb"
	"github.com/ollama/ollama/x/registry/apitype"
	"github.com/ollama/ollama/x/utils/upload"
)

// Defaults
const (
	DefaultUploadChunkSize = 50 * 1024 * 1024
)

type Server struct {
	UploadChunkSize int64 // default is DefaultUploadChunkSize
	S3Client        *minio.Client
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

func (s *Server) uploadChunkSize() int64 {
	return cmp.Or(s.UploadChunkSize, DefaultUploadChunkSize)
}

func (s *Server) handlePush(w http.ResponseWriter, r *http.Request) error {
	const bucketTODO = "test"
	const minimumMultipartSize = 5 * 1024 * 1024 // S3 spec

	pr, err := oweb.DecodeUserJSON[apitype.PushRequest]("", r.Body)
	if err != nil {
		return err
	}

	mp := model.ParseName(pr.Name)
	if !mp.IsComplete() {
		return oweb.Invalid("name", pr.Name, "must be complete")
	}

	m, err := oweb.DecodeUserJSON[apitype.Manifest]("manifest", bytes.NewReader(pr.Manifest))
	if err != nil {
		return err
	}

	mcc := &minio.Core{Client: s.s3()}
	// TODO(bmizerany): complete uploads before stats for any with ETag

	type completeParts struct {
		key   string
		parts []minio.CompletePart
	}

	completePartsByUploadID := make(map[string]completeParts)
	for _, mcp := range pr.CompleteParts {
		// parse the URL
		u, err := url.Parse(mcp.URL)
		if err != nil {
			return err
		}

		q := u.Query()

		// Check if this is a part upload, if not, skip
		uploadID := q.Get("uploadId")
		if uploadID == "" {
			// not a part upload
			continue
		}

		// PartNumber is required
		queryPartNumber := q.Get("partNumber")
		partNumber, err := strconv.Atoi(queryPartNumber)
		if err != nil {
			return oweb.Invalid("partNumber", queryPartNumber, "")
		}
		if partNumber < 1 {
			return oweb.Invalid("partNumber", queryPartNumber, "must be >= 1")
		}

		// ETag is required
		if mcp.ETag == "" {
			return oweb.Missing("etag")
		}

		cp := completePartsByUploadID[uploadID]
		cp.key = u.Path
		cp.parts = append(cp.parts, minio.CompletePart{
			PartNumber: partNumber,
			ETag:       mcp.ETag,
		})
		completePartsByUploadID[uploadID] = cp
	}

	for uploadID, cp := range completePartsByUploadID {
		var zeroOpts minio.PutObjectOptions

		// TODO: gross fix!!!!!!!!!!!!!!!
		key := strings.TrimPrefix(cp.key, "/"+bucketTODO+"/")

		fmt.Printf("Completing multipart upload %s %s %v\n", bucketTODO, key, cp.parts)
		_, err := mcc.CompleteMultipartUpload(r.Context(), bucketTODO, key, uploadID, cp.parts, zeroOpts)
		if err != nil {
			var e minio.ErrorResponse
			if errors.As(err, &e) && e.Code == "NoSuchUpload" {
				return oweb.Invalid("uploadId", uploadID, "")
			}
			return err
		}
	}

	var requirements []apitype.Requirement
	for _, l := range m.Layers {
		// TODO(bmizerany): do in parallel
		if l.Size == 0 {
			continue
		}

		// TODO(bmizerany): "global" throttle of rate of transfer
		pushed, err := s.statObject(r.Context(), l.Digest)
		if err != nil {
			println("ERROR:", "statObject", err)
			return err
		}
		if !pushed {
			key := path.Join("blobs", l.Digest)
			if l.Size < minimumMultipartSize {
				// single part upload
				fmt.Printf("Presigning single %s %s\n", bucketTODO, key)
				signedURL, err := s.s3().PresignedPutObject(r.Context(), bucketTODO, key, 15*time.Minute)
				if err != nil {
					println("ERROR:", "presign single", err)
					return err
				}
				requirements = append(requirements, apitype.Requirement{
					Digest: l.Digest,
					Size:   l.Size,
					URL:    signedURL.String(),
				})
			} else {
				uploadID, err := mcc.NewMultipartUpload(r.Context(), bucketTODO, key, minio.PutObjectOptions{})
				if err != nil {
					return err
				}
				fmt.Printf("Presigning multi %s %s %s\n", bucketTODO, key, uploadID)
				for partNumber, c := range upload.Chunks(l.Size, s.uploadChunkSize()) {
					const timeToStartUpload = 15 * time.Minute

					signedURL, err := s.s3().Presign(r.Context(), "PUT", bucketTODO, key, timeToStartUpload, url.Values{
						"partNumber": []string{strconv.Itoa(partNumber)},
						"uploadId":   []string{uploadID},
					})
					if err != nil {
						println("ERROR:", "presign multi", err)
						return err
					}

					requirements = append(requirements, apitype.Requirement{
						Digest: l.Digest,
						Offset: c.Offset,
						Size:   c.N,
						URL:    signedURL.String(),
					})
				}
			}
		}
	}

	if len(requirements) == 0 {
		// Commit the manifest
		body := bytes.NewReader(pr.Manifest)
		path := path.Join("manifests", path.Join(mp.Parts()...))
		_, err := s.s3().PutObject(r.Context(), bucketTODO, path, body, int64(len(pr.Manifest)), minio.PutObjectOptions{})
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
	_, err = s.s3().StatObject(ctx, "test", path, minio.StatObjectOptions{})
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

func (s *Server) s3() *minio.Client {
	if s.S3Client != nil {
		return s.S3Client
	}
	s3, err := minio.New("localhost:9000", &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin", ""),
		Secure: false,
	})
	if err != nil {
		panic(err)
	}
	return s3
}

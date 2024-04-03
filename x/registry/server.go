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
	"os"
	"path"
	"strconv"
	"time"

	"bllamo.com/build/blob"
	"bllamo.com/client/ollama"
	"bllamo.com/oweb"
	"bllamo.com/registry/apitype"
	"bllamo.com/utils/upload"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// Defaults
const (
	DefaultUploadChunkSize = 50 * 1024 * 1024
)

// TODO(bmizerany): move all env things to package envkobs?
var defaultLibrary = cmp.Or(os.Getenv("OLLAMA_REGISTRY"), "registry.ollama.ai/library")

func DefaultLibrary() string {
	return defaultLibrary
}

type Server struct {
	UploadChunkSize int64 // default is DefaultUploadChunkSize
	minioClient     *minio.Client
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

	ref := blob.ParseRef(pr.Ref)
	if !ref.Complete() {
		return oweb.Mistake("invalid", "name", "must be complete")
	}

	m, err := oweb.DecodeUserJSON[apitype.Manifest]("manifest", bytes.NewReader(pr.Manifest))
	if err != nil {
		return err
	}

	mcc := &minio.Core{Client: s.mc()}
	// TODO(bmizerany): complete uploads before stats for any with ETag

	type completeParts struct {
		key   string
		parts []minio.CompletePart
	}

	completePartsByUploadID := make(map[string]completeParts)
	for _, pu := range pr.Uploaded {
		// parse the URL
		u, err := url.Parse(pu.URL)
		if err != nil {
			return err
		}
		q := u.Query()
		uploadID := q.Get("uploadId")
		if uploadID == "" {
			// not a part upload
			continue
		}
		partNumber, err := strconv.Atoi(q.Get("partNumber"))
		if err != nil {
			return oweb.Mistake("invalid", "url", "invalid or missing PartNumber")
		}
		etag := pu.ETag
		if etag == "" {
			return oweb.Mistake("invalid", "etag", "missing")
		}
		cp, ok := completePartsByUploadID[uploadID]
		if !ok {
			cp = completeParts{key: u.Path}
			completePartsByUploadID[uploadID] = cp
		}
		cp.parts = append(cp.parts, minio.CompletePart{
			PartNumber: partNumber,
			ETag:       etag,
		})
		fmt.Println("uploadID", uploadID, "partNumber", partNumber, "etag", etag)
		completePartsByUploadID[uploadID] = cp
	}

	for uploadID, cp := range completePartsByUploadID {
		var zeroOpts minio.PutObjectOptions
		_, err := mcc.CompleteMultipartUpload(r.Context(), bucketTODO, cp.key, uploadID, cp.parts, zeroOpts)
		if err != nil {
			// log and continue; put backpressure on the client
			log.Printf("error completing upload: %v", err)
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
			return err
		}
		if !pushed {
			key := path.Join("blobs", l.Digest)
			if l.Size < minimumMultipartSize {
				// single part upload
				signedURL, err := s.mc().PresignedPutObject(r.Context(), bucketTODO, key, 15*time.Minute)
				if err != nil {
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
				for partNumber, c := range upload.Chunks(l.Size, s.uploadChunkSize()) {
					const timeToStartUpload = 15 * time.Minute

					signedURL, err := s.mc().Presign(r.Context(), "PUT", bucketTODO, key, timeToStartUpload, url.Values{
						"uploadId":   []string{uploadID},
						"partNumber": []string{strconv.Itoa(partNumber)},
					})
					if err != nil {
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
		path := path.Join("manifests", path.Join(ref.Parts()...))
		_, err := s.mc().PutObject(r.Context(), bucketTODO, path, body, int64(len(pr.Manifest)), minio.PutObjectOptions{})
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

package build

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"

	"bllamo.com/build/blob"
	"bllamo.com/build/internal/blobstore"
	"bllamo.com/model"
)

// Errors
var (
	ErrInvalidRef             = errors.New("invalid ref")
	ErrUnsupportedModelFormat = errors.New("unsupported model format")
	ErrMissingFileType        = errors.New("missing 'general.file_type' key")
	ErrNoSuchBlob             = errors.New("no such blob")
	ErrNotFound               = errors.New("not found")
)

type mediaType string

// Known media types
const (
	mediaTypeModel mediaType = "application/vnd.ollama.image.model"
)

type Server struct {
	st *blobstore.Store
}

// Open starts a new build server that uses dir as the base directory for all
// build artifacts. If dir is empty, DefaultDir is used.
//
// It returns an error if the provided or default dir cannot be initialized.
func Open(dir string) (*Server, error) {
	if dir == "" {
		var err error
		dir, err = DefaultDir()
		if err != nil {
			return nil, err
		}
	}
	st, err := blobstore.Open(dir)
	if err != nil {
		return nil, err
	}
	return &Server{st: st}, nil
}

func (s *Server) Build(ref string, f model.File) error {
	br := blob.ParseRef(ref)
	if !br.Valid() {
		return invalidRef(ref)
	}

	// 1. Resolve FROM
	//   a. If it's a local file (gguf), hash it and add it to the store.
	//   b. If it's a local dir (safetensor), convert to gguf and add to
	//   store.
	//   c. If it's a remote file (http), refuse.
	// 2. Turn other pragmas into layers, and add them to the store.
	// 3. Create a manifest from the layers.
	// 4. Store the manifest in the manifest cache
	// 5. Done.

	if f.From == "" {
		return &model.Error{Pragma: "FROM", Message: "missing"}
	}

	var layers []layerJSON

	id, info, size, err := s.importModel(f.From)
	if err != nil {
		return err
	}
	layers = append(layers, layerJSON{
		ID:        id,
		MediaType: mediaTypeModel,
		Size:      size,
	})

	id, size, err = blobstore.PutString(s.st, f.License)
	if err != nil {
		return err
	}
	layers = append(layers, layerJSON{
		ID:        id,
		MediaType: "text/plain",
		Size:      size,
	})

	data, err := json.Marshal(manifestJSON{Layers: layers})
	if err != nil {
		return err
	}
	return s.st.Set(br.WithBuild(info.FileType.String()), data)
}

func (s *Server) LayerFile(digest string) (string, error) {
	fileName := s.st.OutputFilename(blobstore.ParseID(digest))
	_, err := os.Stat(fileName)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("%w: %q", ErrNoSuchBlob, digest)
	}
	return fileName, nil
}

func (s *Server) Manifest(ref blob.Ref) ([]byte, error) {
	data, _, err := s.getManifestData(ref)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: %q", ErrNotFound, ref)
	}
	return data, err
}

// WeightFile returns the absolute path to the weights file for the given model ref.
func (s *Server) WeightsFile(ref blob.Ref) (string, error) {
	m, err := s.getManifest(ref)
	if err != nil {
		return "", err
	}
	for _, l := range m.Layers {
		if l.MediaType == mediaTypeModel {
			return s.st.OutputFilename(l.ID), nil
		}
	}
	return "", fmt.Errorf("missing weights layer for %q", ref)
}

type manifestJSON struct {
	// Layers is the list of layers in the manifest.
	Layers []layerJSON `json:"layers"`
}

// Layer is a layer in a model manifest.
type layerJSON struct {
	// ID is the ID of the layer.
	ID        blobstore.ID `json:"digest"`
	MediaType mediaType    `json:"mediaType"`
	Size      int64        `json:"size"`
}

func (s *Server) getManifest(ref blob.Ref) (manifestJSON, error) {
	data, path, err := s.getManifestData(ref)
	if err != nil {
		return manifestJSON{}, err
	}
	var m manifestJSON
	if err := json.Unmarshal(data, &m); err != nil {
		return manifestJSON{}, &fs.PathError{Op: "unmarshal", Path: path, Err: err}
	}
	return m, nil
}

func (s *Server) getManifestData(ref blob.Ref) (data []byte, path string, err error) {
	return s.st.Resolve(ref)
}

func invalidRef(ref string) error {
	return fmt.Errorf("%w: %q", ErrInvalidRef, ref)
}

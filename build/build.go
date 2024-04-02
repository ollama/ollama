package build

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"bllamo.com/build/blob"
	"bllamo.com/build/internal/blobstore"
	"bllamo.com/model"
)

// Errors
var (
	ErrIncompleteRef          = errors.New("unqualified ref")
	ErrUnsupportedModelFormat = errors.New("unsupported model format")
	ErrMissingFileType        = errors.New("missing 'general.file_type' key")
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
	if !br.Complete() {
		return fmt.Errorf("%w: %q", ErrIncompleteRef, br.Full())
	}

	// 1. Resolve FROM
	//   a. If it's a local file (gguf), hash it and add it to the store.
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

	data, err := json.Marshal(Manifest{Layers: layers})
	if err != nil {
		return err
	}
	return s.Set(br.WithBuild(info.FileType.String()), data)
}

func (s *Server) LayerFile(digest string) (string, error) {
	fileName := s.st.OutputFilename(blobstore.ParseID(digest))
	_, err := os.Stat(fileName)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("%w: %q", ErrNotFound, digest)
	}
	return fileName, nil
}

func (s *Server) ManifestData(ref string) ([]byte, error) {
	br, err := parseCompleteRef(ref)
	if err != nil {
		return nil, err
	}
	data, _, err := s.resolve(br)
	return data, err
}

// WeightFile returns the absolute path to the weights file for the given model ref.
func (s *Server) WeightsFile(ref string) (string, error) {
	br, err := parseCompleteRef(ref)
	if err != nil {
		return "", err
	}
	m, err := s.getManifest(br)
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

// resolve returns the data for the given ref, if any.
//
// TODO: This should ideally return an ID, but the current on
// disk layout is that the actual manifest is stored in the "ref" instead of
// a pointer to a content-addressed blob. I (bmizerany) think we should
// change the on-disk layout to store the manifest in a content-addressed
// blob, and then have the ref point to that blob. This would simplify the
// code, allow us to have integrity checks on the manifest, and clean up
// this interface.
func (s *Server) resolve(ref blob.Ref) (data []byte, path string, err error) {
	path, err = s.refFileName(ref)
	if err != nil {
		return nil, "", err
	}
	data, err = os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, "", fmt.Errorf("%w: %q", ErrNotFound, ref)
	}
	if err != nil {
		// do not wrap the error here, as it is likely an I/O error
		// and we want to preserve the absraction since we may not
		// be on disk later.
		return nil, "", fmt.Errorf("manifest read error: %v", err)
	}
	return data, path, nil
}

// Set sets the data for the given ref.
func (s *Server) Set(ref blob.Ref, data []byte) error {
	path, err := s.refFileName(ref)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0777); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0666); err != nil {
		return err
	}
	return nil
}

func (s *Server) refFileName(ref blob.Ref) (string, error) {
	if !ref.Complete() {
		return "", fmt.Errorf("ref not fully qualified: %q", ref)
	}
	return filepath.Join(s.st.Dir(), "manifests", filepath.Join(ref.Parts()...)), nil
}

type Manifest struct {
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

func (s *Server) getManifest(ref blob.Ref) (Manifest, error) {
	data, path, err := s.resolve(ref)
	if err != nil {
		return Manifest{}, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, &fs.PathError{Op: "unmarshal", Path: path, Err: err}
	}
	return m, nil
}

func parseCompleteRef(ref string) (blob.Ref, error) {
	br := blob.ParseRef(ref)
	if !br.Complete() {
		return blob.Ref{}, fmt.Errorf("%w: %q", ErrIncompleteRef, ref)
	}
	return br, nil
}

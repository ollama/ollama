package model

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"iter"
	"maps"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

func root() (*os.Root, error) {
	root, err := os.OpenRoot(envconfig.Models())
	if err != nil {
		return nil, err
	}

	for _, sub := range []string{"manifests", "blobs"} {
		if _, err := root.Stat(sub); errors.Is(err, fs.ErrNotExist) {
			if err := root.MkdirAll(sub, 0o750); err != nil {
				return nil, err
			}
		} else if err != nil {
			return nil, err
		}
	}

	return root, nil
}

// Open opens an existing file for reading. It will return [fs.ErrNotExist]
// if the file does not exist. The returned [*Root] can only be used for reading.
// It is the caller's responsibility to close the file when done.
func Open(n Name) (*Root, error) {
	r, err := root()
	if err != nil {
		return nil, err
	}

	f, err := r.Open(filepath.Join("manifests", n.Filepath()))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var m manifest
	if err := json.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}

	blobs := make(map[string]*blob, len(m.Layers)+1)
	blobs[NamePrefix] = m.Config
	for _, layer := range m.Layers {
		if layer.Name == "" && layer.MediaType != "" {
			mediatype, _, err := mime.ParseMediaType(layer.MediaType)
			if err != nil {
				return nil, err
			}

			if suffix, ok := strings.CutPrefix(mediatype, MediaTypePrefix); ok {
				layer.Name = NamePrefix + suffix
			}
		}

		blobs[layer.Name] = layer
	}

	return &Root{
		root:  r,
		name:  n,
		blobs: blobs,
		flags: os.O_RDONLY,
	}, nil
}

// Create creates a new file. The returned [Root] can be used for both reading
// and writing. It is the caller's responsibility to close the file when done
// in order to finalize any new blobs and write the manifest.
func Create(n Name) (*Root, error) {
	r, err := root()
	if err != nil {
		return nil, err
	}

	return &Root{
		root:  r,
		name:  n,
		blobs: make(map[string]*blob),
		flags: os.O_RDWR,
	}, nil
}

type blob struct {
	Digest    string `json:"digest"`
	MediaType string `json:"mediaType"`
	Name      string `json:"name,omitempty"`
	Size      int64  `json:"size"`

	// tempfile is the temporary file where the blob data is written.
	tempfile *os.File

	// hash is the hash.Hash used to compute the blob digest.
	hash hash.Hash
}

func (b *blob) Write(p []byte) (int, error) {
	return io.MultiWriter(b.tempfile, b.hash).Write(p)
}

func (b *blob) Filepath() string {
	return strings.ReplaceAll(b.Digest, ":", "-")
}

type manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        *blob   `json:"config"`
	Layers        []*blob `json:"layers"`
}

// Root represents a model file. It can be used to read and write blobs
// associated with the model.
//
// Blobs are identified by name. Certain names are special and reserved;
// see [NamePrefix] for details.
type Root struct {
	root  *os.Root
	name  Name
	blobs map[string]*blob
	flags int
}

const MediaTypePrefix = "application/vnd.ollama"

// NamePrefix is the prefix used for identifying special names. Names
// with this prefix are idenfitied by their media types:
//
//   - name: NamePrefix + suffix
//   - mediaType: [MediaTypePrefix] + suffix
//
// For example:
//
//   - name: "./..image.model"
//   - mediaType: "application/vnd.ollama.image.model"
//
// NamePrefix by itself identifies the manifest config.
const NamePrefix = "./."

// Open opens the named blob for reading. It is the caller's responsibility
// to close the returned [io.ReadCloser] when done. It will return
// [fs.ErrNotExist] if the blob does not exist.
func (r Root) Open(name string) (io.ReadCloser, error) {
	if b, ok := r.blobs[name]; ok {
		r, err := r.root.Open(filepath.Join("blobs", b.Filepath()))
		if err != nil {
			return nil, err
		}
		return r, nil
	}

	return nil, fs.ErrNotExist
}

func (r Root) ReadFile(name string) ([]byte, error) {
	f, err := r.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return io.ReadAll(f)
}

// Create creates or replaces a named blob in the file. If the blob already
// exists, it will be overwritten. It will return [fs.ErrInvalid] if the file
// was opened in read-only mode. The returned [io.Writer] can be used to write
// to the blob and does not need be closed, but the file must be closed to
// finalize the blob.
func (r *Root) Create(name string) (io.Writer, error) {
	if r.flags&os.O_RDWR != 0 {
		w, err := os.CreateTemp(r.root.Name(), "")
		if err != nil {
			return nil, err
		}

		r.blobs[name] = &blob{Name: name, tempfile: w, hash: sha256.New()}
		return r.blobs[name], nil
	}

	return nil, fs.ErrInvalid
}

// Close closes the file. If the file was opened in read-write mode, it
// will finalize any writeable blobs and write the manifest.
func (r *Root) Close() error {
	if r.flags&os.O_RDWR != 0 {
		for _, b := range r.blobs {
			if b.tempfile != nil {
				fi, err := b.tempfile.Stat()
				if err != nil {
					return err
				}

				if err := b.tempfile.Close(); err != nil {
					return err
				}

				b.Size = fi.Size()
				b.Digest = fmt.Sprintf("sha256:%x", b.hash.Sum(nil))

				if suffix, ok := strings.CutPrefix(b.Name, NamePrefix); ok {
					if b.Name == NamePrefix {
						b.MediaType = "application/vnd.docker.container.image.v1+json"
					} else {
						b.MediaType = MediaTypePrefix + suffix
					}
					b.Name = ""
				}

				rel, err := filepath.Rel(r.root.Name(), b.tempfile.Name())
				if err != nil {
					return err
				}

				if err := r.root.Rename(rel, filepath.Join("blobs", b.Filepath())); err != nil {
					return err
				}
			}
		}

		p := filepath.Join("manifests", r.name.Filepath())
		if _, err := r.root.Stat(filepath.Dir(p)); errors.Is(err, os.ErrNotExist) {
			if err := r.root.MkdirAll(filepath.Dir(p), 0o750); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}

		f, err := r.root.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
		if err != nil {
			return err
		}
		defer f.Close()

		if err := json.NewEncoder(f).Encode(manifest{
			SchemaVersion: 2,
			MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
			Config:        r.blobs[NamePrefix],
			Layers: func() []*blob {
				blobs := make([]*blob, 0, len(r.blobs))
				for name, b := range r.blobs {
					if name != NamePrefix {
						blobs = append(blobs, b)
					}
				}
				return blobs
			}(),
		}); err != nil {
			return err
		}
	}

	return r.root.Close()
}

// Name returns the name of the file.
func (r Root) Name() Name {
	return r.name
}

// Names returns an iterator over the names in the file.
func (r Root) Names() iter.Seq[string] {
	return maps.Keys(r.blobs)
}

// Glob returns an iterator over the names in the file that match the given
// pattern.
//
// The pattern syntax is the same as [filepath.Match]. As with filepath.Match,
// the only possible returned error is ErrBadPattern, when pattern is malformed.
func (r Root) Glob(pattern string) (iter.Seq[string], error) {
	if _, err := filepath.Match(pattern, ""); err != nil {
		return nil, err
	}

	return func(yield func(string) bool) {
		for name, blob := range r.blobs {
			if matched, _ := filepath.Match(pattern, name); matched {
				if !yield(blob.Filepath()) {
					return
				}
			}
		}
	}, nil
}

func (r Root) JoinPath(parts ...string) string {
	return filepath.Join(append([]string{r.root.Name()}, parts...)...)
}

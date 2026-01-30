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
// if the file does not exist. The returned [*File] can only be used for reading.
// It is the caller's responsibility to close the file when done.
func Open(n Name) (*File, error) {
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

	return &File{
		root:  r,
		name:  n,
		blobs: blobs,
		flags: os.O_RDONLY,
	}, nil
}

// Create creates a new file. The returned [File] can be used for both reading
// and writing. It is the caller's responsibility to close the file when done
// in order to finalize any new blobs and write the manifest.
func Create(n Name) (*File, error) {
	r, err := root()
	if err != nil {
		return nil, err
	}

	return &File{
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

// File represents a model file. It can be used to read and write blobs
// associated with the model.
//
// Blobs are identified by name. Certain names are special and reserved;
// see [NamePrefix] for details.
type File struct {
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
func (f File) Open(name string) (io.ReadCloser, error) {
	if b, ok := f.blobs[name]; ok {
		r, err := f.root.Open(filepath.Join("blobs", b.Filepath()))
		if err != nil {
			return nil, err
		}
		return r, nil
	}

	return nil, fs.ErrNotExist
}

// Create creates or replaces a named blob in the file. If the blob already
// exists, it will be overwritten. It will return [fs.ErrInvalid] if the file
// was opened in read-only mode. The returned [io.Writer] can be used to write
// to the blob and does not need be closed, but the file must be closed to
// finalize the blob.
func (f *File) Create(name string) (io.Writer, error) {
	if f.flags&os.O_RDWR != 0 {
		w, err := os.CreateTemp(f.root.Name(), "")
		if err != nil {
			return nil, err
		}

		f.blobs[name] = &blob{Name: name, tempfile: w, hash: sha256.New()}
		return f.blobs[name], nil
	}

	return nil, fs.ErrInvalid
}

// Close closes the file. If the file was opened in read-write mode, it
// will finalize any writeable blobs and write the manifest.
func (f *File) Close() error {
	if f.flags&os.O_RDWR != 0 {
		for _, b := range f.blobs {
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

				rel, err := filepath.Rel(f.root.Name(), b.tempfile.Name())
				if err != nil {
					return err
				}

				if err := f.root.Rename(rel, filepath.Join("blobs", b.Filepath())); err != nil {
					return err
				}
			}
		}

		p := filepath.Join("manifests", f.name.Filepath())
		if _, err := f.root.Stat(filepath.Dir(p)); errors.Is(err, os.ErrNotExist) {
			if err := f.root.MkdirAll(filepath.Dir(p), 0o750); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}

		r, err := f.root.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o640)
		if err != nil {
			return err
		}
		defer r.Close()

		if err := json.NewEncoder(r).Encode(manifest{
			SchemaVersion: 2,
			MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
			Config:        f.blobs[NamePrefix],
			Layers: func() []*blob {
				blobs := make([]*blob, 0, len(f.blobs))
				for name, b := range f.blobs {
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

	return f.root.Close()
}

// Name returns the name of the file.
func (f File) Name() Name {
	return f.name
}

// Names returns an iterator over the names in the file.
func (f File) Names() iter.Seq[string] {
	return maps.Keys(f.blobs)
}

// Glob returns an iterator over the names in the file that match the given
// pattern.
//
// The pattern syntax is the same as [filepath.Match]. As with filepath.Match,
// the only possible returned error is ErrBadPattern, when pattern is malformed.
func (f File) Glob(pattern string) (iter.Seq[string], error) {
	if _, err := filepath.Match(pattern, ""); err != nil {
		return nil, err
	}

	return func(yield func(string) bool) {
		for name := range f.blobs {
			if matched, _ := filepath.Match(pattern, name); matched {
				if !yield(name) {
					return
				}
			}
		}
	}, nil
}

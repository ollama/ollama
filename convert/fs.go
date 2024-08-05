package convert

import (
	"archive/zip"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

type ZipReader struct {
	r *zip.Reader
	p string

	// limit is the maximum size of a file that can be read directly
	// from the zip archive. Files larger than this size will be extracted
	limit int64
}

func NewZipReader(r *zip.Reader, p string, limit int64) fs.FS {
	return &ZipReader{r, p, limit}
}

func (z *ZipReader) Open(name string) (fs.File, error) {
	r, err := z.r.Open(name)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	if fi, err := r.Stat(); err != nil {
		return nil, err
	} else if fi.Size() < z.limit {
		return r, nil
	}

	if !filepath.IsLocal(name) {
		return nil, zip.ErrInsecurePath
	}

	n := filepath.Join(z.p, name)
	if _, err := os.Stat(n); errors.Is(err, os.ErrNotExist) {
		w, err := os.Create(n)
		if err != nil {
			return nil, err
		}
		defer w.Close()

		if _, err := io.Copy(w, r); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}

	return os.Open(n)
}

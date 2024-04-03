package build

import (
	"errors"
	"fmt"
	"os"

	"bllamo.com/build/internal/blobstore"
	"bllamo.com/encoding/gguf"
)

func importError(err error) (blobstore.ID, gguf.Info, int64, error) {
	return blobstore.ID{}, gguf.Info{}, 0, err
}

func (s *Server) importModel(path string) (_ blobstore.ID, _ gguf.Info, size int64, _ error) {
	info, err := os.Stat(path)
	if err != nil {
		return importError(err)
	}
	if info.IsDir() {
		return s.importSafeTensor(path)
	} else {
		return s.importGGUF(path)
	}
}

func (s *Server) importGGUF(path string) (_ blobstore.ID, _ gguf.Info, size int64, _ error) {
	f, err := os.Open(path)
	if err != nil {
		return importError(err)
	}
	defer f.Close()

	info, err := gguf.StatReader(f)
	if errors.Is(err, gguf.ErrBadMagic) {
		return importError(ErrUnsupportedModelFormat)
	}
	if err != nil {
		return importError(err)
	}

	if info.FileType == 0 {
		return importError(fmt.Errorf("%w: %q", ErrMissingFileType, path))
	}
	id, size, err := s.st.Put(f)
	if err != nil {
		return importError(err)
	}
	return id, info, size, nil
}

func (s *Server) importSafeTensor(path string) (_ blobstore.ID, _ gguf.Info, size int64, _ error) {
	path, err := convertSafeTensorToGGUF(path)
	if err != nil {
		return importError(err)
	}
	return s.importGGUF(path)
}

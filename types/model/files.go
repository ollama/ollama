package model

import (
	"io/fs"
	"iter"
	"path/filepath"
)

func All() (iter.Seq[Name], error) {
	r, err := root()
	if err != nil {
		return nil, err
	}

	manifests, err := r.OpenRoot("manifests")
	if err != nil {
		return nil, err
	}

	matches, err := fs.Glob(manifests.FS(), "*/*/*/*")
	if err != nil {
		return nil, err
	}

	return func(yield func(Name) bool) {
		for _, match := range matches {
			name := ParseNameFromFilepath(filepath.ToSlash(match))
			if !yield(name) {
				return
			}
		}
	}, nil
}

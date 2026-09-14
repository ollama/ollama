package convert

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

func parseTorch(fsys fs.FS, replacer *strings.Replacer, ps ...string) ([]Tensor, error) {
	var ts []Tensor
	for _, p := range ps {
		// gopickle's pytorch.Load only accepts an OS filename; it has no
		// fs.FS-aware alternative. Recover the real OS path from fsys rather
		// than passing p through unchanged, which would resolve against the
		// process's working directory instead of fsys.
		osPath, err := resolveOSPath(fsys, p)
		if err != nil {
			return nil, err
		}

		pt, err := pytorch.Load(osPath)
		if err != nil {
			return nil, err
		}

		for _, k := range pt.(*types.Dict).Keys() {
			t := pt.(*types.Dict).MustGet(k)

			var shape []uint64
			for dim := range t.(*pytorch.Tensor).Size {
				shape = append(shape, uint64(dim))
			}

			ts = append(ts, torch{
				storage: t.(*pytorch.Tensor).Source,
				tensorBase: &tensorBase{
					name:  replacer.Replace(k.(string)),
					shape: shape,
				},
			})
		}
	}

	return ts, nil
}

// resolveOSPath recovers the real OS filename backing name in fsys. It only
// works for OS-backed filesystems (e.g. os.DirFS, the only kind ConvertModel
// passes in), since fsys.Open returns the concrete *os.File in that case.
func resolveOSPath(fsys fs.FS, name string) (string, error) {
	f, err := fsys.Open(name)
	if err != nil {
		return "", err
	}
	defer f.Close()

	osFile, ok := f.(*os.File)
	if !ok {
		return "", fmt.Errorf("%s: torch format requires an OS-backed filesystem", name)
	}

	return osFile.Name(), nil
}

type torch struct {
	storage pytorch.StorageInterface
	*tensorBase
}

func (t torch) Clone() Tensor {
	return torch{
		storage: t.storage,
		tensorBase: &tensorBase{
			name:     t.name,
			shape:    t.shape,
			repacker: t.repacker,
		},
	}
}

func (pt torch) WriteTo(w io.Writer) (int64, error) {
	return 0, nil
}

package convert

import (
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
		f, err := fsys.Open(p)
		if err != nil {
			return nil, err
		}

		filePath := p
		if osFile, ok := f.(*os.File); ok {
			filePath = osFile.Name()
			_ = f.Close()
		} else {
			tmp, err := os.CreateTemp("", "ollama-torch-*.bin")
			if err != nil {
				_ = f.Close()
				return nil, err
			}
			defer os.Remove(tmp.Name())
			if _, err := io.Copy(tmp, f); err != nil {
				_ = f.Close()
				_ = tmp.Close()
				return nil, err
			}
			_ = f.Close()
			_ = tmp.Close()
			filePath = tmp.Name()
		}

		pt, err := pytorch.Load(filePath)
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

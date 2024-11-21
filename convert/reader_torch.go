package convert

import (
	"io"
	"io/fs"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

func parseTorch(fsys fs.FS, replacer *strings.Replacer, ps ...string) ([]Tensor, error) {
	var ts []Tensor
	for _, p := range ps {
		pt, err := pytorch.Load(p)
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

func (pt torch) WriteTo(w io.Writer) (int64, error) {
	return 0, nil
}

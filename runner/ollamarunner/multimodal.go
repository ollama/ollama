package ollamarunner

import (
	"errors"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// Tensors can't be used across multiple compute graphs. This is a problem
// if a single embedding is split across batches using views since all of
// the views will have the same source tensor. We also don't want to
// recompute the entire embedding for each batch.
//
// To avoid this, we compute all of the tensors for the embedding on the
// first use and then store the result in system memory. When we need
// additional tensors, we recreate them from the stored data.

// multimodalEntry represents the embeddings of a single object (such
// as an image).
type multimodalEntry struct {
	// mm is the original set of tensors created by EncodeMultimodal
	mm []input.Multimodal

	// data is the computed result of mm. Nil if not yet computed
	data [][]float32
}

// multimodalStore maps from an individual tensor (of which there
// may be many in a single multimodal object) to its parent embedding
type multimodalStore map[ml.Tensor]*multimodalEntry

func newMultimodalStore() multimodalStore {
	return make(multimodalStore)
}

// addMultimodal stores an embedding for later use in a compute graph
func (m multimodalStore) addMultimodal(embedding []input.Multimodal) {
	entry := &multimodalEntry{mm: embedding}

	for _, e := range embedding {
		if e.Tensor != nil {
			m[e.Tensor] = entry
		}
	}
}

// getMultimodal takes a source set of tensors (which may contain a whole or
// parts of one or more images) and returns the equivalent that can be used in
// the current context
func (m multimodalStore) getMultimodal(backend ml.Backend, ctx ml.Context, in []input.Multimodal, reserve bool) ([]input.Multimodal, error) {
	out := make([]input.Multimodal, len(in))
	for i := range out {
		if in[i].Tensor != nil {
			var err error
			out[i].Tensor, err = m.getTensor(backend, ctx, in[i].Tensor, reserve)
			if err != nil {
				return nil, err
			}
		}

		out[i].Data = in[i].Data
	}

	return out, nil
}

func (m multimodalStore) getTensor(backend ml.Backend, ctx ml.Context, in ml.Tensor, reserve bool) (ml.Tensor, error) {
	entry := m[in]

	if entry.data == nil {
		computeCtx := backend.NewContext()
		defer computeCtx.Close()

		var tensors []ml.Tensor
		for _, t := range entry.mm {
			if t.Tensor != nil {
				tensors = append(tensors, t.Tensor)
			}
		}

		if len(tensors) == 0 {
			return nil, nil
		}

		computeCtx.Forward(tensors...)
		entry.data = make([][]float32, len(entry.mm))

		if !reserve {
			computeCtx.Compute(tensors...)

			for i, t := range entry.mm {
				if t.Tensor != nil {
					entry.data[i] = t.Tensor.Floats()
				}
			}
		} else {
			computeCtx.Reserve()
		}
	}

	for i, t := range entry.mm {
		if in == t.Tensor {
			if !reserve {
				return ctx.Input().FromFloatSlice(entry.data[i], t.Tensor.Shape()...)
			} else {
				return ctx.Input().Empty(t.Tensor.DType(), t.Tensor.Shape()...), nil
			}
		}
	}

	return nil, errors.New("multimodal tensor not found")
}

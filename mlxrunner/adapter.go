package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lora"
)

type adapterModel interface {
	WithAdapter(map[string]*lora.Factors) model.Model
}

// A view shares the base arrays, tokenizer and sampler. Only its factors are
// owned here. Slots keep the view alive for both requests and cached prefixes.
type adapterView struct {
	ref     manifest.AdapterRef
	runner  *Runner
	weights *mlx.Scope
	slots   int
}

func (slot *batchSlot) runner(base *Runner) *Runner {
	if slot != nil && slot.adapter != nil {
		return slot.adapter.runner
	}
	return base
}

func (slot *batchSlot) adapterRef() manifest.AdapterRef {
	if slot != nil && slot.adapter != nil {
		return slot.adapter.ref
	}
	return manifest.AdapterRef{}
}

func requestAdapter(req queuedRequest) manifest.AdapterRef {
	switch req := req.(type) {
	case request:
		return req.Adapter
	case *evaluationRequest:
		return req.adapter
	default:
		return manifest.AdapterRef{}
	}
}

func (slot *batchSlot) releaseAdapter() {
	if view := slot.adapter; view != nil {
		view.slots--
		if view.slots == 0 {
			view.weights.Close()
		}
		slot.adapter = nil
	}
}

// prepareSlot binds an idle slot before prefix lookup. On a cold load,
// reclaim its previous view and budget for the incoming factors before eval.
func (s *batchScheduler) prepareSlot(index int, pending queuedRequest) error {
	ref := requestAdapter(pending)
	old := s.slots[index]
	if old.adapterRef() == ref {
		s.trimSnapshots(pending, index, 0)
		return nil
	}

	var view *adapterView
	var checkpoint *lora.Checkpoint
	var reserve int64
	if ref != (manifest.AdapterRef{}) {
		for _, slot := range s.slots {
			if slot.adapterRef() == ref {
				view = slot.adapter
				break
			}
		}
		if view == nil {
			var err error
			checkpoint, err = s.r.inspectAdapter(ref)
			if err != nil {
				return err
			}
			reserve = checkpoint.Bytes
		}
	}

	if old != nil {
		old.cache.close()
		old.scoreCache.close()
		old.releaseAdapter()
		s.slots[index] = nil
	}
	s.trimSnapshots(pending, index, reserve)
	if checkpoint != nil {
		var err error
		view, err = s.r.loadAdapter(ref, checkpoint)
		if err != nil {
			return err
		}
	}
	if view != nil {
		view.slots++
	}
	s.slots[index] = &batchSlot{id: index, adapter: view}
	return nil
}

func (r *Runner) inspectAdapter(ref manifest.AdapterRef) (*lora.Checkpoint, error) {
	if _, ok := r.Model.(adapterModel); !ok {
		return nil, fmt.Errorf("model does not support MLX LoRA")
	}
	data, err := r.baseManifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	layout, err := lora.ModelLayout(data)
	if err != nil {
		return nil, err
	}
	return lora.Inspect(ref, layout)
}

func (r *Runner) loadAdapter(ref manifest.AdapterRef, checkpoint *lora.Checkpoint) (*adapterView, error) {
	factors, arrays, err := checkpoint.Load()
	if err != nil {
		return nil, err
	}
	mlx.Eval(arrays...)
	weights := mlx.NewScope()
	weights.Attach(arrays...)
	// Adapted forwards use plain decoding. The publisher's draft can still
	// serve base-only requests from the same process.
	view := &Runner{
		Model:   r.Model.(adapterModel).WithAdapter(factors),
		Sampler: r.Sampler, Tokenizer: r.Tokenizer, grammarEngine: r.grammarEngine,
		contextLength: r.contextLength, parallel: r.parallel,
	}
	return &adapterView{ref: ref, runner: view, weights: weights}, nil
}

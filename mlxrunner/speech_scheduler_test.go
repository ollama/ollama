package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

type speechBatchTestModel struct {
	stubModel
	batches [][]float32
}

type speechBatchTestState struct {
	value  float32
	steps  int
	closed int
	err    error
}

func (s *speechBatchTestState) Close() { s.closed++ }

func (m *speechBatchTestModel) NewSpeech(_ context.Context, req llm.SpeechRequest) (model.SpeechState, error) {
	return &speechBatchTestState{value: float32(req.Input[0]-'0') / 10}, nil
}

func (m *speechBatchTestModel) DecodeSpeech(states []model.SpeechState) ([]model.SpeechOutput, error) {
	values := make([]float32, len(states))
	out := make([]model.SpeechOutput, len(states))
	for i, state := range states {
		s := state.(*speechBatchTestState)
		s.steps++
		values[i] = s.value
		out[i] = model.SpeechOutput{Samples: []float32{s.value}, Done: s.steps == 3, Err: s.err}
	}
	m.batches = append(m.batches, values)
	return out, nil
}

func TestSpeechBatchCancellationAndReuse(t *testing.T) {
	ctx := t.Context()
	for _, failure := range []string{"cancel", "model-error"} {
		mlxtest.RunSubtest(t, failure, func(t *mlxtest.T) {
			m := &speechBatchTestModel{}
			s := &batchScheduler{r: &Runner{Model: m}, slots: make([]*batchSlot, 2)}
			cancelCtx, cancel := context.WithCancel(ctx)
			defer cancel()
			var requests []*speechRequest
			for _, input := range []string{"1", "2", "3"} {
				requests = append(requests, &speechRequest{ctx: ctx, input: llm.SpeechRequest{Input: input}, queue: &audioQueue{ready: make(chan struct{}, 1)}, done: make(chan error, 1)})
			}
			requests[0].ctx = cancelCtx
			s.admit(requests[0])
			s.admit(requests[1])
			s.step()
			wantErr := context.Canceled
			if failure == "cancel" {
				cancel()
			} else {
				wantErr = errors.New("bad speech frame")
				requests[0].state.(*speechBatchTestState).err = wantErr
			}
			s.step()
			if err := <-requests[0].done; !errors.Is(err, wantErr) {
				t.Fatalf("request error = %v, want %v", err, wantErr)
			}
			if requests[0].state.(*speechBatchTestState).closed != 1 {
				t.Fatal("error acknowledgement preceded cleanup")
			}
			s.admit(requests[2])
			for range 3 {
				s.step()
			}
			for i, request := range requests {
				if i > 0 {
					if err := <-request.done; err != nil {
						t.Fatal(err)
					}
					request.queue.close()
					chunks, _ := request.queue.next()
					if !bytes.Equal(bytes.Join(chunks, nil), bytes.Repeat(pcm16([]float32{float32(i+1) / 10}), 3)) {
						t.Fatalf("stream %d mixed samples or lost frames", i)
					}
				}
				if request.state.(*speechBatchTestState).closed != 1 {
					t.Fatalf("stream %d cleanup count != 1", i)
				}
				select {
				case err := <-request.done:
					t.Fatalf("duplicate completion: %v", err)
				default:
				}
			}
			if !slices.Equal(m.batches[0], []float32{.1, .2}) || !slices.Equal(m.batches[2], []float32{.3, .2}) {
				t.Fatalf("batch membership = %v", m.batches)
			}
			for _, slot := range s.slots {
				if slot.busy() {
					t.Fatal("finished speech still owns a slot")
				}
			}
		})
	}
}

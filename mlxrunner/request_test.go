package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlx/mlxthread"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/sample"
)

type mixedRequestModel struct {
	schedulingModel
	speech speechBatchTestModel
}

func (m *mixedRequestModel) NewSpeech(ctx context.Context, input llm.SpeechRequest) (model.SpeechState, error) {
	return m.speech.NewSpeech(ctx, input)
}

func (m *mixedRequestModel) DecodeSpeech(states []model.SpeechState) ([]model.SpeechOutput, error) {
	return m.speech.DecodeSpeech(states)
}

func TestSchedulerMixedRequests(t *testing.T) {
	ctx := t.Context()
	for _, parallel := range []int{1, 2} {
		for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
			mlxtest.RunSubtest(t, fmt.Sprintf("parallel%d/order%v", parallel, order), func(t *mlxtest.T) {
				tok := newTestTokenizer(t, []int32{7})
				m := &mixedRequestModel{schedulingModel: schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}}
				// Buffer the supplied order so the test can inspect admission before
				// advancing the worker. Production senders wait on this same channel.
				r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: parallel, Requests: make(chan queuedRequest, 3)}
				s := &batchScheduler{r: r, slots: make([]*batchSlot, parallel)}
				defer func() {
					for _, slot := range s.slots {
						if slot != nil {
							if slot.cache != nil {
								slot.cache.close()
							}
							if slot.scoreCache != nil {
								slot.scoreCache.close()
							}
						}
					}
					r.Sampler.Free()
				}()
				generation := request{Ctx: ctx, Tokens: []int32{1, 1, 1}, Responses: make(chan CompletionResponse, 16), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 8}}}
				var embedding []float32
				evaluations := 0
				evaluation := &evaluationRequest{ctx: ctx, done: make(chan error, 1), open: func(slot *batchSlot) (*evaluationState, error) {
					return r.openEvaluation(slot, &model.PreparedRequest{Tokens: []int32{2, 2}}, nil, func(hidden *mlx.Array) error {
						evaluations++
						embedding = meanPoolAndNormalize(hidden.Reshape(2, mtpTestVocab), 2)
						return nil
					}), nil
				}}
				speech := &speechRequest{ctx: ctx, input: llm.SpeechRequest{Input: "3"}, queue: &audioQueue{ready: make(chan struct{}, 1)}, done: make(chan error, 1)}
				requests := []queuedRequest{generation, evaluation, speech}
				for _, i := range order {
					r.Requests <- requests[i]
				}
				s.admitPending()
				if len(r.Requests) != len(requests)-parallel {
					t.Fatal("admission exceeded the shared slot limit")
				}
				for i, slot := range s.slots {
					switch want := requests[order[i]].(type) {
					case request:
						if slot.request == nil || slot.request.request.Responses != want.Responses {
							t.Fatal("generation was admitted out of order")
						}
					case *evaluationRequest:
						if slot.evaluation == nil || slot.evaluation.request != want {
							t.Fatal("evaluation was admitted out of order")
						}
					case *speechRequest:
						if slot.speech != want {
							t.Fatal("speech was admitted out of order")
						}
					}
				}
				for range 24 {
					mlx.Scoped(func() {
						s.admitPending()
						s.step()
					})
				}
				for _, slot := range s.slots {
					if slot.busy() {
						t.Fatal("mixed requests did not all finish")
					}
				}
				if len(r.Requests) != 0 {
					t.Fatal("a request remained queued")
				}
				var text strings.Builder
				var final CompletionResponse
				for response := range generation.Responses {
					if response.Error != nil {
						t.Fatal(response.Error)
					}
					text.WriteString(response.Content)
					final = response
				}
				if !final.Done || final.EvalCount != 8 || text.String() != strings.Repeat(tok.Decode([]int32{1}), 8) {
					t.Fatalf("generation mixed results: %q, %+v", text.String(), final)
				}
				if err := <-evaluation.done; err != nil {
					t.Fatal(err)
				}
				if evaluations != 1 || !slices.Equal(embedding, []float32{0, 0, 1, 0, 0, 0, 0, 0}) {
					t.Fatalf("evaluation count %d, embedding %v", evaluations, embedding)
				}
				if err := <-speech.done; err != nil {
					t.Fatal(err)
				}
				speech.queue.close()
				chunks, _ := speech.queue.next()
				if !bytes.Equal(bytes.Join(chunks, nil), bytes.Repeat(pcm16([]float32{.3}), 3)) || speech.state.(*speechBatchTestState).closed != 1 {
					t.Fatal("speech mixed samples, lost frames or was not closed once")
				}
			})
		}
	}
}

// A request removed from admission still belongs to the scheduler if the MLX
// worker rejects dispatch. Every request type must release its waiting caller.
func TestSchedulerRejectsUnadmittedRequests(t *testing.T) {
	for _, reason := range []string{"canceled", "stopped-worker"} {
		for _, kind := range []string{"generation", "evaluation", "speech"} {
			t.Run(reason+"/"+kind, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()
				requestCtx, cancelRequest := context.WithCancel(ctx)
				defer cancelRequest()
				responses := make(chan CompletionResponse, 1)
				completed := make(chan error, 1)
				grammar := &grammarCompilation{done: make(chan struct{})}
				var pending queuedRequest
				switch kind {
				case "generation":
					pending = request{Ctx: requestCtx, Responses: responses, Grammar: grammar}
				case "evaluation":
					pending = &evaluationRequest{ctx: requestCtx, done: completed}
				case "speech":
					pending = &speechRequest{ctx: requestCtx, done: completed}
				}
				wantErr := context.Canceled
				if reason == "canceled" {
					cancelRequest()
					s := &batchScheduler{r: &Runner{}, slots: make([]*batchSlot, 1)}
					s.admit(pending)
					if s.slots[0] != nil {
						t.Fatal("canceled request acquired a slot")
					}
				} else {
					worker, err := mlxthread.Start("queue-stopped", nil)
					if err != nil {
						t.Fatal(err)
					}
					if err := worker.Stop(ctx, nil); err != nil {
						t.Fatal(err)
					}
					wantErr = mlxthread.ErrStopped
					r := &Runner{parallel: 1, Requests: make(chan queuedRequest), mlxThread: worker}
					stopped := make(chan error, 1)
					go func() { stopped <- r.runBatchRequests(ctx) }()
					select {
					case r.Requests <- pending:
					case <-ctx.Done():
						t.Fatal("scheduler did not receive request")
					}
					select {
					case err := <-stopped:
						if !errors.Is(err, wantErr) {
							t.Fatalf("scheduler error = %v, want %v", err, wantErr)
						}
					case <-ctx.Done():
						t.Fatal("scheduler did not exit")
					}
				}
				if kind == "generation" {
					select {
					case response, open := <-responses:
						if reason == "canceled" {
							if open {
								t.Fatalf("canceled generation produced a response: %+v", response)
							}
						} else {
							if !open || response.Error == nil || response.Error.StatusCode != http.StatusInternalServerError || response.Error.ErrorMessage != wantErr.Error() {
								t.Fatalf("rejected generation response = %+v, want internal error %v", response, wantErr)
							}
							if _, open := <-responses; open {
								t.Fatal("rejected generation left its response stream open")
							}
						}
						if !grammar.abandoned {
							t.Fatal("rejected generation retained its grammar")
						}
					default:
						t.Fatal("rejected generation left its caller waiting")
					}
				} else {
					select {
					case err := <-completed:
						if !errors.Is(err, wantErr) {
							t.Fatalf("completion error = %v, want %v", err, wantErr)
						}
					default:
						t.Fatal("rejected request left its caller waiting")
					}
				}
			})
		}
	}
}

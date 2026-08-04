package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestRepeatGuard(t *testing.T) {
	// how many times a unit has to be emitted to spend the budget, with room
	// for the turns that go into recognising the cycle in the first place
	repeats := func(unit ...string) int {
		n := 0
		for _, s := range unit {
			n += len(s)
		}
		return repeatGuardBudgetBytes/n + 8
	}

	tests := []struct {
		name  string
		emit  func(g *repeatGuard) bool
		trips bool
	}{
		{
			// the tail of an mp3 is a long run of one byte, so the tail of its
			// base64 is a long run of one token
			name: "a base64 payload is not a runaway",
			emit: func(g *repeatGuard) bool {
				return emitAll(g, strings.Split("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR", ""))
			},
		},
		{
			name: "a long run of one token below the budget is allowed",
			emit: func(g *repeatGuard) bool {
				return emitTimes(g, 1000, "VVVV")
			},
		},
		{
			name: "a run of one token past the budget stops",
			emit: func(g *repeatGuard) bool {
				return emitTimes(g, repeats("VVVV"), "VVVV")
			},
			trips: true,
		},
		{
			// the shape a stuck model actually produces, which counting a
			// single repeated token never caught
			name: "a repeated phrase past the budget stops",
			emit: func(g *repeatGuard) bool {
				return emitTimes(g, repeats("Wait", ",", " let", " me", " re", "-read", " the", " question", "."),
					"Wait", ",", " let", " me", " re", "-read", " the", " question", ".")
			},
			trips: true,
		},
		{
			name: "indentation does not stop generation",
			emit: func(g *repeatGuard) bool {
				for range 200 {
					if g.observe("\n") || g.observe("    ") || g.observe("if") || g.observe(" x") {
						return true
					}
				}
				return false
			},
		},
		{
			name: "ordinary prose does not stop generation",
			emit: func(g *repeatGuard) bool {
				words := strings.Fields("the quick brown fox jumps over the lazy dog and then the dog barks back")
				for i := range 500 {
					// the same words in a different order each time, which is
					// what prose looks like and a loop does not
					for j := range words {
						if g.observe(words[(i+j*7)%len(words)]) {
							return true
						}
					}
				}
				return false
			},
		},
		{
			// a payload that repeats and then moves on must not carry its
			// count into whatever follows
			name: "a run that ends resets the budget",
			emit: func(g *repeatGuard) bool {
				if emitTimes(g, repeats("VVVV")-16, "VVVV") {
					return true
				}
				emitAll(g, strings.Fields("and that is the end of the file"))
				return emitTimes(g, repeats("AAAA")-16, "AAAA")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var g repeatGuard
			if got := tt.emit(&g); got != tt.trips {
				t.Fatalf("tripped = %v, want %v", got, tt.trips)
			}
		})
	}
}

// emitAll feeds a sequence to the guard once, reporting whether it tripped.
func emitAll(g *repeatGuard, tokens []string) bool {
	for _, tok := range tokens {
		if g.observe(tok) {
			return true
		}
	}
	return false
}

// emitTimes feeds a unit to the guard n times, reporting whether it tripped.
func emitTimes(g *repeatGuard, n int, unit ...string) bool {
	for range n {
		if emitAll(g, unit) {
			return true
		}
	}
	return false
}

func TestLlamaServerCompletionRepeatedOutput(t *testing.T) {
	tests := []struct {
		name           string
		emits          int
		wantDoneReason DoneReason
		wantContent    string
	}{
		{
			// what reading an mp3 back as base64 looks like on the wire
			name:           "a repetitive payload completes normally",
			emits:          1000,
			wantDoneReason: DoneReasonStop,
			wantContent:    strings.Repeat("VVVV", 1000) + "done",
		},
		{
			name:           "a runaway is stopped and reported",
			emits:          repeatGuardBudgetBytes/len("VVVV") + 10,
			wantDoneReason: DoneReasonRepeat,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for range tt.emits {
					fmt.Fprintln(w, `data: {"content":"VVVV","stop":false}`)
				}
				fmt.Fprintln(w, `data: {"content":"done","stop":true,"stop_type":"eos","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
			}))
			defer srv.Close()

			var portInt int
			fmt.Sscanf(srv.URL[strings.LastIndex(srv.URL, ":")+1:], "%d", &portInt)

			runner := &llamaServerRunner{
				port:    portInt,
				cmd:     fakeRunningCmd(),
				sem:     semaphore.NewWeighted(1),
				options: api.Options{Runner: api.Runner{NumCtx: 2048}},
			}

			opts := api.DefaultOptions()
			var content strings.Builder
			var final CompletionResponse
			var sawDone bool
			err := runner.Completion(t.Context(), CompletionRequest{Prompt: "read the file", Options: &opts}, func(cr CompletionResponse) {
				content.WriteString(cr.Content)
				if cr.Done {
					final = cr
					sawDone = true
				}
			})
			if err != nil {
				t.Fatalf("Completion error: %v", err)
			}

			// a caller that never sees Done cannot tell a finished stream from
			// a broken connection, which is the whole point of this test
			if !sawDone {
				t.Fatal("no done response was sent")
			}
			if final.DoneReason != tt.wantDoneReason {
				t.Errorf("done reason = %q, want %q", final.DoneReason, tt.wantDoneReason)
			}
			if tt.wantContent != "" && content.String() != tt.wantContent {
				t.Errorf("content = %d chars, want %d", content.Len(), len(tt.wantContent))
			}
		})
	}
}

package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestClientScore(t *testing.T) {
	input := llm.ScoreRequest{MaxTokens: 2048, Rows: []llm.ScoreRow{{Prompt: "prompt", Candidates: []string{"A", "B"}}}}
	for _, status := range []int{http.StatusOK, http.StatusBadRequest} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/score" || r.Method != "POST" {
				t.Errorf("wrong endpoint: %s %s", r.Method, r.URL.Path)
			}
			var got llm.ScoreRequest
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil || !reflect.DeepEqual(got, input) {
				t.Errorf("request changed: %+v, %v", got, err)
			}
			if status != http.StatusOK {
				http.Error(w, "prompt too long", status)
				return
			}
			_ = json.NewEncoder(w).Encode(llm.ScoreResponse{Logits: [][]float32{{-1, 2}}, InputTokens: 3})
		}))
		client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
		result, err := client.Score(context.Background(), input)
		srv.Close()
		if status == http.StatusOK {
			if err != nil || result.InputTokens != 3 || !reflect.DeepEqual(result.Logits, [][]float32{{-1, 2}}) {
				t.Fatalf("unexpected result: %+v, %v", result, err)
			}
		} else {
			var statusErr api.StatusError
			if !errors.As(err, &statusErr) || statusErr.StatusCode != status || statusErr.ErrorMessage != "prompt too long" {
				t.Fatalf("status error not preserved: %v", err)
			}
		}
	}
}

func TestClientScoreTransportError(t *testing.T) {
	for _, message := range []string{"", "MLX: failed to allocate memory"} {
		t.Run(message, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				conn.Close()
			}))
			defer srv.Close()
			status := &llm.StatusWriter{}
			status.SetLastError(message)
			client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client(), status: status}
			_, err := client.Score(t.Context(), llm.ScoreRequest{})
			if message == "" {
				if !errors.Is(err, io.EOF) {
					t.Fatalf("transport error lost: %v", err)
				}
			} else if !llm.IsOutOfMemory(err) || !strings.Contains(err.Error(), message) {
				t.Fatalf("runner OOM diagnostic lost: %v", err)
			}
		})
	}
}

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"golang.org/x/sync/semaphore"
)

func TestLlamaServerSystemOne(t *testing.T) {
	var request decision.Request
	if err := json.Unmarshal([]byte(`{"model":"test-model","state":{"z":1e2,"a":"state"},"questions":{"q":{"type":"noul","instructions":"Present?","criteria":{}},"choice":{"type":"choice","instructions":{"rule":"Route it"},"criteria":{"z":null,"a":"first"}},"score":{"type":"score","instructions":"Rate it","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}`), &request); err != nil {
		t.Fatal(err)
	}
	request.Images = []api.ImageData{{137, 80, 78, 71, 13, 10, 26, 10}}
	const response = `{"answers":{"q":{"type":"noul","noul":0.75},"choice":{"type":"choice","choice":"z","probabilities":{"z":0.8,"a":0.2},"confidence":0.6},"score":{"type":"score","score":1.2,"legend":{"0":"low","1":"mid","2":"high"},"probabilities":{"0":0.1,"1":0.6,"2":0.3},"confidence":0.4}},"usage":{"input_tokens":29,"output_tokens":0}}`
	var want decision.Response
	if err := json.Unmarshal([]byte(response), &want); err != nil {
		t.Fatal(err)
	}
	want.Model = request.Model
	for _, status := range []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			calls := 0
			mux := http.NewServeMux()
			mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"status":"ok"}`) })
			mux.HandleFunc("/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
				calls++
				var input struct {
					decision.Request
					Images []string `json:"images"`
				}
				if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
					t.Error(err)
				}
				if len(input.Images) != 1 || input.Images[0] != "data:image/png;base64,iVBORw0KGgo=" {
					t.Errorf("wrong image transport: %v", input.Images)
				}
				if r.Method != http.MethodPost || !reflect.DeepEqual(input.State, request.State) || !reflect.DeepEqual(input.Questions, request.Questions) {
					t.Errorf("unexpected input: %s %+v", r.Method, input)
				}
				w.WriteHeader(status)
				if status != http.StatusOK {
					fmt.Fprint(w, `{"error":{"message":"invalid decision","type":"invalid_request_error"}}`)
					return
				}
				fmt.Fprint(w, response)
			})
			srv := httptest.NewServer(mux)
			defer srv.Close()
			runner := &llamaServerRunner{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client(), cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1)}
			result, err := runner.SystemOne(t.Context(), request)
			if status == http.StatusOK {
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(result, want) {
					t.Fatalf("upstream answer changed: got %+v, want %+v", result, want)
				}
			} else {
				var statusErr api.StatusError
				if !errors.As(err, &statusErr) || statusErr.StatusCode != status {
					t.Fatalf("status %d returned %v", status, err)
				}
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := runner.SystemOne(ctx, decision.Request{}); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled request: %v", err)
			}
			if calls != 1 {
				t.Fatalf("canceled request reached runner, calls=%d", calls)
			}
			if !runner.sem.TryAcquire(1) {
				t.Fatal("request leaked semaphore permit")
			}
			runner.sem.Release(1)
		})
	}
}

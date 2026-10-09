package llm

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestLlamaServerSystemOne(t *testing.T) {
	var got struct {
		State     json.RawMessage `json:"state"`
		Images    []string        `json:"images"`
		Questions json.RawMessage `json:"questions"`
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		fmt.Fprint(w, `{"model":"x","answers":{"ok":{"type":"noul","noul":0.75}},"usage":{"input_tokens":7,"output_tokens":0}}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	runner := &llamaServerRunner{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client(), cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1)}

	state, questions := json.RawMessage(`{"z":1.0,"a":"x"}`), json.RawMessage(`{"ok":{"type":"noul","instructions":"Fine?"}}`)
	png := api.ImageData{137, 80, 78, 71, 13, 10, 26, 10}
	answers, usage, err := runner.SystemOne(t.Context(), state, questions, []api.ImageData{png})
	if err != nil {
		t.Fatal(err)
	}
	if string(got.State) != string(state) || string(got.Questions) != string(questions) || len(got.Images) != 1 || !strings.HasPrefix(got.Images[0], "data:image/png;base64,") {
		t.Fatalf("request changed on the way to llama-server: %+v", got)
	}
	if string(answers) != `{"ok":{"type":"noul","noul":0.75}}` || string(usage) != `{"input_tokens":7,"output_tokens":0}` {
		t.Fatalf("answers = %s, usage = %s", answers, usage)
	}
}

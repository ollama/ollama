package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// disconnectRecorder is like the responseRecorder in routes_create_test.go, but
// its CloseNotify channel is controllable so a test can simulate the client
// going away mid-stream.
type disconnectRecorder struct {
	*httptest.ResponseRecorder
	closeCh chan bool
}

func newDisconnectRecorder() *disconnectRecorder {
	return &disconnectRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		closeCh:          make(chan bool),
	}
}

func (r *disconnectRecorder) CloseNotify() <-chan bool {
	return r.closeCh
}

// TestGenerateStreamDisconnectDoesNotLeakGoroutine is a regression test: when
// a streaming /api/generate client disconnects mid-response, the goroutine
// driving the completion must not block forever trying to send into the now
// unread response channel.
//
// Real disconnects surface through two signals that fire together in Go's
// net/http server: the ResponseWriter's CloseNotify channel (what gin's
// Context.Stream reacts to, to stop calling its step function) and the
// request's context being cancelled (what net/http's connection handling
// cancels once the underlying connection goes away, independently of
// whether the handler ever calls CloseNotify). Both are simulated here
// since routes.go's fix relies on the latter.
func TestGenerateStreamDisconnectDoesNotLeakGoroutine(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	gin.SetMode(gin.TestMode)

	const extraChunksAfterDisconnect = 5

	firstChunkSent := make(chan struct{})
	clientDisconnected := make(chan struct{})
	completionReturned := make(chan struct{})

	mock := &mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			defer close(completionReturned)

			// First chunk: lets the test know it can now simulate the client
			// going away.
			fn(llm.CompletionResponse{Content: "hello ", Done: false})
			close(firstChunkSent)

			<-clientDisconnected

			// Deliberately keep calling fn without checking ctx here, the
			// same way a real completion loop does between tokens - the
			// protection against a leaked/blocked goroutine has to live in
			// how routes.go sends into the channel, not in this mock.
			for range extraChunksAfterDisconnect {
				fn(llm.CompletionResponse{Content: "more ", Done: false})
			}

			fn(llm.CompletionResponse{
				Content:    "done",
				Done:       true,
				DoneReason: llm.DoneReasonStop,
			})
			return nil
		},
	}

	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "test", nil, "{{ .Prompt }}", nil)

	streamTrue := true
	reqBody, err := json.Marshal(api.GenerateRequest{
		Model:  "test",
		Prompt: "hi",
		Stream: &streamTrue,
	})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx, cancelReq := context.WithCancel(context.Background())
	defer cancelReq()

	rec := newDisconnectRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = (&http.Request{
		Method: http.MethodPost,
		Body:   io.NopCloser(bytes.NewReader(reqBody)),
	}).WithContext(reqCtx)

	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		s.GenerateHandler(c)
	}()

	select {
	case <-firstChunkSent:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for the first streamed chunk")
	}

	// Simulate the client disconnecting. In Go's real net/http server these
	// two things happen together as soon as the underlying connection goes
	// away: the ResponseWriter's CloseNotify fires (which is what gin's
	// Context.Stream reacts to) and the request's context is cancelled
	// (which is what the fix in routes.go relies on to stop blocked sends).
	close(rec.closeCh)
	cancelReq()

	select {
	case <-handlerDone:
	case <-time.After(5 * time.Second):
		t.Fatal("GenerateHandler did not return after the client disconnected " +
			"(streamResponse/c.Stream should exit as soon as CloseNotify fires)")
	}

	// Let the mock's completion loop run the rest of the way: since none of
	// its sends after the disconnect have a reader anymore, an unguarded
	// `ch <- res` in routes.go would block here forever, and this would time
	// out instead of the completion function ever returning.
	close(clientDisconnected)

	select {
	case <-completionReturned:
	case <-time.After(5 * time.Second):
		t.Fatal("the completion goroutine appears to be leaked: it never " +
			"finished sending the remaining post-disconnect chunks")
	}
}

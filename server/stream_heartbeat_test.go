package server

import (
	"bufio"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func parseGenerateStream(t *testing.T, body string) []api.GenerateResponse {
	t.Helper()

	var events []api.GenerateResponse
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event api.GenerateResponse
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("failed to decode stream line %q: %v", line, err)
		}
		events = append(events, event)
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("failed to scan stream body: %v", err)
	}

	return events
}

func TestStreamResponseWithHeartbeat_EmitsDuringSilence(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("GET", "/", nil)

	ch := make(chan any)
	done := make(chan struct{})
	go func() {
		streamResponseWithOptions(c, ch, streamResponseOptions{
			heartbeatInterval: 15 * time.Millisecond,
			heartbeatValue:    generateHeartbeatValue("test"),
		})
		close(done)
	}()

	time.Sleep(55 * time.Millisecond)
	ch <- api.GenerateResponse{
		Model:     "test",
		CreatedAt: time.Now().UTC(),
		Response:  "hello",
		Done:      false,
	}
	ch <- api.GenerateResponse{
		Model:      "test",
		CreatedAt:  time.Now().UTC(),
		Response:   "",
		Done:       true,
		DoneReason: "stop",
	}
	close(ch)
	<-done

	events := parseGenerateStream(t, rec.Body.String())
	var keepalives int
	var final int
	for _, event := range events {
		if !event.Done && event.Response == "" {
			keepalives++
		}
		if event.Done {
			final++
		}
	}

	if keepalives == 0 {
		t.Fatalf("expected at least one heartbeat event, got %d events", len(events))
	}
	if final != 1 {
		t.Fatalf("expected exactly one final done event, got %d", final)
	}
}

func TestStreamResponseWithHeartbeat_NoKeepaliveWhenFrequent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("GET", "/", nil)

	ch := make(chan any)
	done := make(chan struct{})
	go func() {
		streamResponseWithOptions(c, ch, streamResponseOptions{
			heartbeatInterval: 40 * time.Millisecond,
			heartbeatValue:    generateHeartbeatValue("test"),
		})
		close(done)
	}()

	for i := 0; i < 3; i++ {
		ch <- api.GenerateResponse{
			Model:     "test",
			CreatedAt: time.Now().UTC(),
			Response:  "x",
			Done:      false,
		}
		time.Sleep(10 * time.Millisecond)
	}
	ch <- api.GenerateResponse{
		Model:      "test",
		CreatedAt:  time.Now().UTC(),
		Response:   "",
		Done:       true,
		DoneReason: "stop",
	}
	close(ch)
	<-done

	events := parseGenerateStream(t, rec.Body.String())
	for _, event := range events {
		if !event.Done && event.Response == "" {
			t.Fatalf("did not expect heartbeat event when output is frequent: %+v", event)
		}
	}
}

func TestStreamResponseWithHeartbeat_Disabled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("GET", "/", nil)

	ch := make(chan any)
	done := make(chan struct{})
	go func() {
		streamResponseWithOptions(c, ch, streamResponseOptions{
			heartbeatInterval: 0,
			heartbeatValue:    generateHeartbeatValue("test"),
		})
		close(done)
	}()

	time.Sleep(50 * time.Millisecond)
	ch <- api.GenerateResponse{
		Model:     "test",
		CreatedAt: time.Now().UTC(),
		Response:  "hello",
		Done:      false,
	}
	ch <- api.GenerateResponse{
		Model:      "test",
		CreatedAt:  time.Now().UTC(),
		Response:   "",
		Done:       true,
		DoneReason: "stop",
	}
	close(ch)
	<-done

	events := parseGenerateStream(t, rec.Body.String())
	var keepalives int
	var finals int
	for _, event := range events {
		if !event.Done && event.Response == "" {
			keepalives++
		}
		if event.Done {
			finals++
		}
	}

	if keepalives != 0 {
		t.Fatalf("expected no keepalive events when disabled, got %d", keepalives)
	}
	if finals != 1 {
		t.Fatalf("expected exactly one final done event, got %d", finals)
	}
}

func TestStreamHeartbeatInterval(t *testing.T) {
	t.Setenv("OLLAMA_STREAM_HEARTBEAT_MS", "10000")

	// default from env
	if got := streamHeartbeatInterval(nil); got != 10*time.Second {
		t.Fatalf("expected default heartbeat 10s, got %s", got)
	}

	// request-level override wins
	override := 25
	if got := streamHeartbeatInterval(&api.StreamOptions{HeartbeatMS: &override}); got != 25*time.Millisecond {
		t.Fatalf("expected override heartbeat 25ms, got %s", got)
	}

	// request-level disable
	disable := 0
	if got := streamHeartbeatInterval(&api.StreamOptions{HeartbeatMS: &disable}); got != 0 {
		t.Fatalf("expected disabled heartbeat, got %s", got)
	}
}

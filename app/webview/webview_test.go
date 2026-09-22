//go:build windows || darwin

package webview

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAsyncBindingDoesNotBlockOtherBindings(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	finished := make(chan string, 1)
	slow := &binding{
		async: true,
		call: func(id, req string) (interface{}, error) {
			close(started)
			<-release
			return id + req, nil
		},
	}
	returned := make(chan struct{})
	go func() {
		slow.invoke("first", "[]", func(_ int, result string) { finished <- result })
		close(returned)
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("background binding did not start")
	}
	select {
	case <-returned:
	case <-time.After(time.Second):
		t.Fatal("binding held the caller while discovery was pending")
	}
	fast := &binding{call: func(_, _ string) (interface{}, error) { return "ready", nil }}
	var result string
	fast.invoke("second", "[]", func(_ int, value string) { result = value })
	if result != `"ready"` {
		t.Fatalf("synchronous binding did not finish on its caller: %q", result)
	}
	select {
	case <-finished:
		t.Fatal("slow binding finished before it was released")
	default:
	}
}

func TestAsyncBindingReturnsResultsAndErrors(t *testing.T) {
	for _, tt := range []struct {
		name   string
		value  interface{}
		err    error
		status int
		want   string
	}{
		{"value", []string{"saved", "cloud"}, nil, 0, `["saved","cloud"]`},
		{"error", nil, errors.New("lookup failed"), -1, `"lookup failed"`},
		{"encoding error", make(chan int), nil, -1, `json: unsupported type`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			replies := make(chan struct {
				status int
				result string
			}, 1)
			b := &binding{async: true, call: func(id, req string) (interface{}, error) {
				if id != "document:1" || req != "[]" {
					return nil, errors.New("callback arguments changed")
				}
				return tt.value, tt.err
			}}
			b.invoke("document:1", "[]", func(status int, result string) {
				replies <- struct {
					status int
					result string
				}{status, result}
			})
			select {
			case reply := <-replies:
				if reply.status != tt.status || !strings.Contains(reply.result, tt.want) {
					t.Fatalf("reply = (%d, %s), want (%d, %s)", reply.status, reply.result, tt.status, tt.want)
				}
			case <-time.After(time.Second):
				t.Fatal("binding did not deliver a result")
			}
		})
	}
}

func TestDestroyDiscardsPendingBindingReplies(t *testing.T) {
	w := &webview{bindings: map[string]uintptr{}, pending: map[uintptr]struct{}{}}
	m.Lock()
	bindingID, dispatchID := index, index+1
	index += 2
	w.bindings["models"] = bindingID
	w.pending[dispatchID] = struct{}{}
	bindings[bindingID] = &binding{owner: w}
	dispatch[dispatchID] = func() { t.Error("closed view received a reply") }
	m.Unlock()
	w.Destroy()
	// A worker completing after destruction must not call into the native view.
	w.returnAsync("models", bindingID, "document:1", 0, `[]`)
	w.Destroy()
	m.Lock()
	defer m.Unlock()
	if bindings[bindingID] != nil || dispatch[dispatchID] != nil || len(w.pending) != 0 {
		t.Fatal("destroyed view retained a binding or pending reply")
	}
}

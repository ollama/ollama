package chat

import (
	"context"
	"testing"
)

func TestCloudAuthTickDoesNotPoll(t *testing.T) {
	polls := 0
	m := chatModel{
		cloudAuthPrompt: &cloudAuthPrompt{polling: true},
		opts: Options{
			PollCloudAuth: func(context.Context) (string, bool) {
				polls++
				return "", false
			},
		},
	}

	updated, cmd := m.updateCloudAuthPrompt(cloudAuthTickMsg{})
	m = updated.(chatModel)

	if m.cloudAuthPrompt.spinner != 1 {
		t.Fatalf("spinner = %d, want 1", m.cloudAuthPrompt.spinner)
	}
	if polls != 0 {
		t.Fatalf("polls = %d, want 0 before running returned tick command", polls)
	}
	if cmd == nil {
		t.Fatal("tick should schedule the next tick")
	}
	if _, ok := cmd().(cloudAuthTickMsg); !ok {
		t.Fatal("tick should schedule another tick, not a poll")
	}
	if polls != 0 {
		t.Fatalf("polls = %d, want 0 after running returned tick command", polls)
	}
}

func TestCloudAuthPollSchedulesNextPoll(t *testing.T) {
	polls := 0
	m := chatModel{
		cloudAuthPrompt: &cloudAuthPrompt{polling: true},
		opts: Options{
			PollCloudAuth: func(context.Context) (string, bool) {
				polls++
				return "", false
			},
		},
	}

	_, cmd := m.updateCloudAuthPrompt(cloudAuthPollMsg{})
	if cmd == nil {
		t.Fatal("poll should schedule the next poll")
	}
	msg, ok := cmd().(cloudAuthPollMsg)
	if !ok {
		t.Fatal("poll should schedule another poll, not a tick")
	}
	if msg.done {
		t.Fatal("poll should report not done")
	}
	if polls != 1 {
		t.Fatalf("polls = %d, want 1", polls)
	}
}

package chat

import (
	"context"
	"errors"
	"strings"
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

func TestCloudModelPreflightFailureShowsPlanVerificationNotice(t *testing.T) {
	m := chatModel{
		opts: Options{
			Model: "glm-5.2:cloud",
		},
	}

	updated, cmd := m.updateCloudModelPreflight(cloudModelPreflightMsg{
		model: "glm-5.2:cloud",
		err:   errors.New("temporary network failure"),
	})
	if cmd != nil {
		t.Fatal("transient preflight failure should not start an auth modal")
	}
	m = updated.(chatModel)

	if got := m.status; got != "Could not verify Ollama plan" {
		t.Fatalf("status = %q", got)
	}
	if m.cloudAuthPrompt != nil {
		t.Fatalf("cloud auth prompt = %#v, want nil", m.cloudAuthPrompt)
	}
}

func TestCloudModelPreflightIgnoresStaleModel(t *testing.T) {
	m := chatModel{
		opts: Options{
			Model: "glm-5.2:cloud",
		},
		status: "ready",
	}

	updated, _ := m.updateCloudModelPreflight(cloudModelPreflightMsg{
		model: "kimi-k2.7-code:cloud",
		err:   errors.New("temporary network failure"),
	})
	m = updated.(chatModel)

	if got := m.status; got != "ready" {
		t.Fatalf("status = %q, want unchanged", got)
	}
}

func TestCloudModelPreflightCommandChecksCloudModel(t *testing.T) {
	var checkedModel, checkedPlan string
	cmd := cloudModelPreflightCmd(context.Background(), Options{
		CheckCloudModel: func(_ context.Context, model, requiredPlan string) error {
			checkedModel = model
			checkedPlan = requiredPlan
			return errors.New("temporary network failure")
		},
		ModelOptions: func(context.Context) ([]ModelOption, error) {
			return []ModelOption{{Name: "glm-5.2:cloud", RequiredPlan: "pro", Cloud: true}}, nil
		},
	}, "glm-5.2:cloud", "")
	if cmd == nil {
		t.Fatal("cloud preflight command should be scheduled")
	}
	raw := cmd()
	msg, ok := raw.(cloudModelPreflightMsg)
	if !ok {
		t.Fatalf("message = %T, want cloudModelPreflightMsg", raw)
	}
	if checkedModel != "glm-5.2:cloud" || checkedPlan != "pro" {
		t.Fatalf("checked model/plan = %q/%q", checkedModel, checkedPlan)
	}
	if msg.model != "glm-5.2:cloud" || msg.err == nil || !strings.Contains(msg.err.Error(), "temporary") {
		t.Fatalf("message = %#v", msg)
	}
}

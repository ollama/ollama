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
			PollCloudAuth: func(context.Context) (string, bool, error) {
				polls++
				return "", false, nil
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
			PollCloudAuth: func(context.Context) (string, bool, error) {
				polls++
				return "", false, nil
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

	if got := m.status; got != cloudPlanVerificationUnavailable {
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

func TestCloudAuthPollGivesUpAfterConsecutiveFailures(t *testing.T) {
	pollErr := errors.New("whoami: connection refused")
	m := chatModel{
		cloudAuthPrompt: &cloudAuthPrompt{polling: true, kind: cloudAuthSignIn},
		opts: Options{
			PollCloudAuth: func(context.Context) (string, bool, error) {
				return "", false, pollErr
			},
		},
	}

	// The first maxPollFailures-1 failures should keep retrying.
	for i := 1; i < maxPollFailures; i++ {
		updated, _ := m.updateCloudAuthPrompt(cloudAuthPollMsg{done: false, err: pollErr})
		m = updated.(chatModel)
		if m.cloudAuthPrompt == nil {
			t.Fatalf("failure %d: prompt cleared early", i)
		}
		if got := m.cloudAuthPrompt.pollFailures; got != i {
			t.Fatalf("failure %d: pollFailures = %d, want %d", i, got, i)
		}
		if m.cloudAuthPrompt.pollErr != pollErr.Error() {
			t.Fatalf("failure %d: pollErr = %q, want %q", i, m.cloudAuthPrompt.pollErr, pollErr.Error())
		}
	}

	// The threshold failure gives up: prompt cleared, back to ready, error entry.
	updated, cmd := m.updateCloudAuthPrompt(cloudAuthPollMsg{done: false, err: pollErr})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatalf("threshold failure should not reschedule, got cmd %T", cmd)
	}
	if m.cloudAuthPrompt != nil {
		t.Fatalf("prompt = %#v, want nil after give-up", m.cloudAuthPrompt)
	}
	if m.status != "ready" {
		t.Fatalf("status = %q, want ready", m.status)
	}
	if len(m.entries) == 0 {
		t.Fatal("expected an error entry after give-up")
	}
	last := m.entries[len(m.entries)-1]
	if last.role != "error" || !strings.Contains(last.content, "couldn't verify sign-in") {
		t.Fatalf("last entry = %+v, want error containing sign-in failure", last)
	}
}

func TestCloudAuthPollResetsFailuresOnHealthyResponse(t *testing.T) {
	pollErr := errors.New("whoami: timeout")
	m := chatModel{
		cloudAuthPrompt: &cloudAuthPrompt{polling: true, kind: cloudAuthSignIn},
		opts: Options{
			PollCloudAuth: func(context.Context) (string, bool, error) {
				return "", false, pollErr
			},
		},
	}

	// Accumulate some failures without hitting the threshold.
	for range maxPollFailures - 2 {
		updated, _ := m.updateCloudAuthPrompt(cloudAuthPollMsg{done: false, err: pollErr})
		m = updated.(chatModel)
	}
	if got := m.cloudAuthPrompt.pollFailures; got != maxPollFailures-2 {
		t.Fatalf("pollFailures = %d, want %d", got, maxPollFailures-2)
	}

	// A healthy (no-error, not-done) response resets the streak so a later
	// transient blip isn't counted against a recovered connection.
	updated, _ := m.updateCloudAuthPrompt(cloudAuthPollMsg{done: false, err: nil})
	m = updated.(chatModel)
	if m.cloudAuthPrompt == nil {
		t.Fatal("healthy response should keep the prompt open")
	}
	if got := m.cloudAuthPrompt.pollFailures; got != 0 {
		t.Fatalf("pollFailures = %d, want 0 after healthy response", got)
	}
	if m.cloudAuthPrompt.pollErr != "" {
		t.Fatalf("pollErr = %q, want empty after healthy response", m.cloudAuthPrompt.pollErr)
	}
}

func TestCloudAuthPollCompletesAfterFailures(t *testing.T) {
	pollErr := errors.New("whoami: timeout")
	m := chatModel{
		cloudAuthPrompt: &cloudAuthPrompt{
			modelName:    "glm-5.2:cloud",
			polling:      true,
			kind:         cloudAuthSignIn,
			pollFailures: maxPollFailures - 1,
		},
		opts: Options{
			CheckCloudModel: func(context.Context, string, string) error { return nil },
			PollCloudAuth:   func(context.Context) (string, bool, error) { return "", false, pollErr },
		},
	}

	// A successful sign-in mid-retry should clear the failure state and re-check.
	updated, _ := m.updateCloudAuthPrompt(cloudAuthPollMsg{done: true})
	m = updated.(chatModel)
	if m.cloudAuthPrompt.polling {
		t.Fatal("done should stop polling")
	}
	if m.cloudAuthPrompt.pollFailures != 0 || m.cloudAuthPrompt.pollErr != "" {
		t.Fatalf("failure state not reset: failures=%d err=%q", m.cloudAuthPrompt.pollFailures, m.cloudAuthPrompt.pollErr)
	}
}

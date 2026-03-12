package launch

import (
	"strings"
	"testing"
)

func TestErrCancelled(t *testing.T) {
	t.Run("NotNil", func(t *testing.T) {
		if errCancelled == nil {
			t.Error("errCancelled should not be nil")
		}
	})

	t.Run("Message", func(t *testing.T) {
		if errCancelled.Error() != "cancelled" {
			t.Errorf("expected 'cancelled', got %q", errCancelled.Error())
		}
	})
}

func TestWithLaunchConfirmPolicy_ChainsAndRestores(t *testing.T) {
	oldPolicy := currentLaunchConfirmPolicy
	oldHook := DefaultConfirmPrompt
	t.Cleanup(func() {
		currentLaunchConfirmPolicy = oldPolicy
		DefaultConfirmPrompt = oldHook
	})

	currentLaunchConfirmPolicy = launchConfirmPolicy{}
	var hookCalls int
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		hookCalls++
		return true, nil
	}

	restoreOuter := withLaunchConfirmPolicy(launchConfirmPolicy{requireBypassMessage: true})
	restoreInner := withLaunchConfirmPolicy(launchConfirmPolicy{bypass: true})

	ok, err := ConfirmPrompt("test prompt")
	if err != nil {
		t.Fatalf("expected bypass policy to allow prompt, got error: %v", err)
	}
	if !ok {
		t.Fatal("expected bypass policy to auto-accept prompt")
	}
	if hookCalls != 0 {
		t.Fatalf("expected bypass to skip hook, got %d hook calls", hookCalls)
	}

	restoreInner()

	_, err = ConfirmPrompt("test prompt")
	if err == nil {
		t.Fatal("expected requireBypassMessage policy to block prompt")
	}
	if !strings.Contains(err.Error(), "re-run with --bypass") {
		t.Fatalf("expected actionable bypass error, got: %v", err)
	}
	if hookCalls != 0 {
		t.Fatalf("expected blocking policy to skip hook, got %d hook calls", hookCalls)
	}

	restoreOuter()

	ok, err = ConfirmPrompt("test prompt")
	if err != nil {
		t.Fatalf("expected restored default behavior to use hook, got error: %v", err)
	}
	if !ok {
		t.Fatal("expected hook to return true")
	}
	if hookCalls != 1 {
		t.Fatalf("expected one hook call after restore, got %d", hookCalls)
	}
}

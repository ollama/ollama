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

func TestWithLaunchConfirmPolicy_ScopesAndRestores(t *testing.T) {
	oldPolicy := currentLaunchConfirmPolicy
	oldHook := DefaultConfirmPrompt
	t.Cleanup(func() {
		currentLaunchConfirmPolicy = oldPolicy
		DefaultConfirmPrompt = oldHook
	})

	currentLaunchConfirmPolicy = launchConfirmPolicy{}
	var hookCalls int
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		hookCalls++
		return true, nil
	}

	restoreOuter := withLaunchConfirmPolicy(launchConfirmPolicy{requireYesMessage: true})
	restoreInner := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})

	ok, err := ConfirmPrompt("test prompt")
	if err != nil {
		t.Fatalf("expected --yes policy to allow prompt, got error: %v", err)
	}
	if !ok {
		t.Fatal("expected --yes policy to auto-accept prompt")
	}
	if hookCalls != 0 {
		t.Fatalf("expected --yes to skip hook, got %d hook calls", hookCalls)
	}

	restoreInner()

	_, err = ConfirmPrompt("test prompt")
	if err == nil {
		t.Fatal("expected requireYesMessage policy to block prompt")
	}
	if !strings.Contains(err.Error(), "re-run with --yes") {
		t.Fatalf("expected actionable --yes error, got: %v", err)
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

func TestConfirmPromptWithOptions_DelegatesToOptionsHook(t *testing.T) {
	oldPolicy := currentLaunchConfirmPolicy
	oldHook := DefaultConfirmPrompt
	t.Cleanup(func() {
		currentLaunchConfirmPolicy = oldPolicy
		DefaultConfirmPrompt = oldHook
	})

	currentLaunchConfirmPolicy = launchConfirmPolicy{}
	called := false
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		called = true
		if prompt != "Connect now?" {
			t.Fatalf("unexpected prompt: %q", prompt)
		}
		if options.YesLabel != "Yes" || options.NoLabel != "Set up later" {
			t.Fatalf("unexpected options: %+v", options)
		}
		return true, nil
	}

	ok, err := ConfirmPromptWithOptions("Connect now?", ConfirmOptions{
		YesLabel: "Yes",
		NoLabel:  "Set up later",
	})
	if err != nil {
		t.Fatalf("ConfirmPromptWithOptions() error = %v", err)
	}
	if !ok {
		t.Fatal("expected confirm to return true")
	}
	if !called {
		t.Fatal("expected options hook to be called")
	}
}

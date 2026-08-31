package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"

	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
)

// for testing
var (
	isInteractiveTerminal = func() bool {
		return term.IsTerminal(int(os.Stdin.Fd())) && term.IsTerminal(int(os.Stdout.Fd()))
	}

	confirmCloudSuggestion = func(prompt string) (bool, error) {
		// Zero-value options default to Yes being preselected.
		return launch.ConfirmPromptWithOptions(prompt, launch.ConfirmOptions{})
	}
)

// pullModelNotFoundMessage is how a registry 404 during pull surfaces to
// clients: os.ErrNotExist wrapped server-side and flattened into the error
// string of the pull stream.
const pullModelNotFoundMessage = "pull model manifest: file does not exist"

// isPullNotFoundErr reports whether err is a pull failure caused by the
// requested model or tag not existing in the registry.
func isPullNotFoundErr(err error) bool {
	return err != nil && strings.Contains(err.Error(), pullModelNotFoundMessage)
}

// cloudSuggestionCandidate reports whether a failed pull of name should
// trigger a ":cloud" suggestion, and if so returns the cloud model name to
// suggest. It only applies to default-tag lookups (e.g. "kimi-k3") against
// the default registry whose pull failed because the tag doesn't exist.
func cloudSuggestionCandidate(name string, pullErr error, insecure bool) (string, bool) {
	if !isPullNotFoundErr(pullErr) {
		return "", false
	}
	return cloudSuggestionName(name, insecure)
}

// cloudSuggestionName applies the name-based eligibility checks for the
// ":cloud" suggestion, returning the cloud model name to suggest.
func cloudSuggestionName(name string, insecure bool) (string, bool) {
	// --insecure implies a non-default registry, where an ollama.com cloud
	// model wouldn't be a meaningful suggestion.
	if insecure {
		return "", false
	}

	ref, err := modelref.ParseRef(name)
	if err != nil || ref.Source != modelref.ModelSourceUnspecified {
		return "", false
	}

	if modelref.HasExplicitTag(ref.Base) {
		return "", false
	}

	// Only default-registry names qualify: the existence probe forwards the name
	// to ollama.com, and custom-registry model names shouldn't be sent there.
	if n := model.ParseName(ref.Base); !n.IsValid() || !strings.EqualFold(n.Host, model.DefaultName().Host) {
		return "", false
	}

	return ref.Base + ":cloud", true
}

// cloudSuggestionRetryCommand renders the command hinted at in
// non-interactive errors for the plain CLI verbs ("run" and "pull").
func cloudSuggestionRetryCommand(verb string) func(string) string {
	return func(cloudName string) string {
		return "ollama " + verb + " " + cloudName
	}
}

// pullWithCloudSuggestion pulls `name`, and if the model's default tag
// doesn't exist but a ":cloud" tag does, offers it: either interactively via
// a confirmation prompt, or by augmenting the returned error when not at a
// terminal. It returns the name that was actually pulled. `retryCommand`
// renders the command suggested in the non-interactive hint for a given
// cloud model name.
func pullWithCloudSuggestion(ctx context.Context, client *api.Client, name string, insecure bool, retryCommand func(cloudName string) string) (string, error) {
	// If a suggestion prompt may follow a failed pull, erase the failed
	// attempt's progress display instead of leaving its "pulling manifest"
	// line to stack up against the accepted pull's identical one.
	_, eligible := cloudSuggestionName(name, insecure)
	clearNotFound := eligible && isInteractiveTerminal()

	pullErr := pullModelWithProgress(ctx, client, name, insecure, clearNotFound)
	if pullErr == nil {
		return name, nil
	}

	cloudName, ok := cloudSuggestionCandidate(name, pullErr, insecure)
	if !ok || ctx.Err() != nil {
		return "", pullErr
	}

	// Showing a ":cloud" model is proxied to ollama.com and mirrors its status,
	// so this reliably answers "does a cloud version exist?". Any error (no
	// cloud tag, cloud disabled, older server, offline) means no suggestion.
	if _, err := client.Show(ctx, &api.ShowRequest{Model: cloudName}); err != nil {
		return "", pullErr
	}

	if !isInteractiveTerminal() {
		if retryCommand == nil {
			return "", fmt.Errorf("%w\n\n%q is available as a cloud model", pullErr, cloudName)
		}
		return "", fmt.Errorf("%w\n\n%q is available as a cloud model. Try:\n  %s", pullErr, cloudName, retryCommand(cloudName))
	}

	accepted, err := confirmCloudSuggestion(fmt.Sprintf("Did you mean %q?", cloudName))
	if err != nil || !accepted {
		// Declining or cancelling falls back to the original error.
		return "", pullErr
	}

	if err := pullModelWithProgress(ctx, client, cloudName, insecure, false); err != nil {
		return "", err
	}
	return cloudName, nil
}

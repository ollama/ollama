package launch

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func (c *launcherClient) localServerContextLength(ctx context.Context) (int, bool) {
	if c == nil || c.apiClient == nil {
		return 0, false
	}
	status, err := c.apiClient.CloudStatusExperimental(ctx)
	if err != nil || status.ContextLength <= 0 {
		return 0, false
	}
	return status.ContextLength, true
}

func confirmLocalContextWarning(integration string, current, recommended int) error {
	shortWarning := fmt.Sprintf(
		"Warning: %s works best with at least %s context; current local context is %s.",
		integration,
		formatContextLength(recommended),
		formatContextLength(current),
	)
	if currentLaunchConfirmPolicy.yes {
		fmt.Fprintf(os.Stderr, "%s\nContinuing because --yes was provided.\n", shortWarning)
		return nil
	}
	if currentLaunchConfirmPolicy.requireYesMessage {
		return fmt.Errorf("%s Re-run with --yes to continue", shortWarning)
	}

	ok, err := ConfirmPromptWithOptions(localContextLengthPrompt(integration, current, recommended), ConfirmOptions{
		YesLabel: "Launch anyway",
		NoLabel:  "Cancel",
		Default:  ConfirmDefaultNo,
	})
	if err != nil {
		return err
	}
	if !ok {
		return ErrCancelled
	}
	return nil
}

func localContextLengthPrompt(integration string, current, recommended int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s works best with at least %s context. ", integration, formatContextLength(recommended))
	fmt.Fprintf(&b, "Current local context: %s. ", formatContextLength(current))
	b.WriteString("Adjust Context length in Ollama Settings and restart to change this:\n")
	b.WriteString("  https://docs.ollama.com/context-length")
	fmt.Fprintf(&b, "\n\nLaunch %s anyway?", integration)
	return b.String()
}

func formatContextLength(tokens int) string {
	switch {
	case tokens <= 0:
		return strconv.Itoa(tokens)
	case tokens%1024 == 0:
		return strconv.Itoa(tokens/1024) + "k"
	case tokens%1000 == 0:
		return strconv.Itoa(tokens/1000) + "k"
	default:
		return strconv.Itoa(tokens)
	}
}

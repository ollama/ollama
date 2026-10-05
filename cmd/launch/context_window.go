package launch

import (
	"context"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/modelref"
)

// LoadedContextWindow reports the context length model is currently running
// with, per the server's process list — the size the scheduler actually
// allocated, which VRAM fit or server configuration may hold below the
// model's trained maximum. Returns 0 when it cannot be determined.
func LoadedContextWindow(ctx context.Context, client *api.Client, model string) int {
	if client == nil || strings.TrimSpace(model) == "" {
		return 0
	}
	resp, err := client.ListRunning(ctx)
	if err != nil {
		return 0
	}
	return processContextWindow(model, resp)
}

func processContextWindow(model string, resp *api.ProcessResponse) int {
	if resp == nil {
		return 0
	}
	for _, running := range resp.Models {
		if running.ContextLength <= 0 {
			continue
		}
		if SameModelRef(model, running.Name) || SameModelRef(model, running.Model) {
			return running.ContextLength
		}
	}
	return 0
}

// SameModelRef reports whether two references name the same model, tolerating
// an explicit ":latest" tag and an unspecified source on either side.
func SameModelRef(a, b string) bool {
	a = comparableModelRef(a)
	b = comparableModelRef(b)
	if strings.EqualFold(a, b) {
		return true
	}
	pa, errA := modelref.ParseRef(a)
	pb, errB := modelref.ParseRef(b)
	if errA != nil || errB != nil {
		return false
	}
	if !strings.EqualFold(pa.Base, pb.Base) {
		return false
	}
	return pa.Source == pb.Source ||
		pa.Source == modelref.ModelSourceUnspecified ||
		pb.Source == modelref.ModelSourceUnspecified
}

func comparableModelRef(value string) string {
	value = strings.TrimSpace(value)
	if strings.HasSuffix(strings.ToLower(value), ":latest") {
		return strings.TrimSpace(value[:len(value)-len(":latest")])
	}
	return value
}

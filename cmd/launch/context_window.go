package launch

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/modelref"
)

const launchContextLoadTimeout = 5 * time.Minute

type contextWindowSource string

const (
	contextWindowSourceRuntime   contextWindowSource = "loaded runtime"
	contextWindowSourceParameter contextWindowSource = "model num_ctx parameter"
	contextWindowSourceNative    contextWindowSource = "native model metadata"
	contextWindowSourceInventory contextWindowSource = "model inventory"
	contextWindowSourceCloud     contextWindowSource = "cloud model metadata"
)

// contextWindowResolution keeps the runner-verified context separate from
// metadata fallbacks. Callers can safely continue with ContextLength while
// still telling users when the server's effective value could not be checked.
type contextWindowResolution struct {
	ContextLength   int
	RuntimeVerified bool
	Source          contextWindowSource
	Err             error
}

func resolveLaunchContextFromEnvironment(model LaunchModel, load bool) contextWindowResolution {
	if model.Remote || isCloudModelName(model.Name) {
		return resolveLaunchContext(context.Background(), nil, model.WithCloudLimits(), false)
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return contextWindowFallback(model, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), launchContextLoadTimeout)
	defer cancel()
	return resolveLaunchContext(ctx, client, model, load)
}

// resolveLaunchContext obtains the context the local server actually assigned
// from /api/ps. Actual launches first issue Ollama's model-only generation
// request, which loads without changing any model options. Configuration-only
// calls skip that load and use an already-running process or metadata.
func resolveLaunchContext(ctx context.Context, client *api.Client, model LaunchModel, load bool) contextWindowResolution {
	if model.Remote || isCloudModelName(model.Name) {
		model = model.WithCloudLimits()
		return contextWindowResolution{ContextLength: launchModelContextLength(model), Source: contextWindowSourceCloud}
	}

	var failures []error
	if client != nil {
		if load {
			if err := client.Generate(ctx, &api.GenerateRequest{Model: model.Name}, func(api.GenerateResponse) error { return nil }); err != nil {
				failures = append(failures, fmt.Errorf("load model: %w", err))
			}
		}

		if running, err := client.ListRunning(ctx); err == nil {
			if n := processContextWindow(model.Name, running); n > 0 {
				return contextWindowResolution{ContextLength: n, RuntimeVerified: true, Source: contextWindowSourceRuntime}
			}
			failures = append(failures, errors.New("model was not present in the running process list"))
		} else {
			failures = append(failures, fmt.Errorf("read running models: %w", err))
		}

		if shown, err := client.Show(ctx, &api.ShowRequest{Model: model.Name}); err == nil {
			native, _ := modelInfoContextLength(shown.ModelInfo)
			if configured, ok := modelParameterNumCtx(shown.Parameters); ok {
				if native > 0 && configured > native {
					configured = native
				}
				return contextWindowResolution{ContextLength: configured, Source: contextWindowSourceParameter, Err: errors.Join(failures...)}
			}
			if native > 0 {
				return contextWindowResolution{ContextLength: native, Source: contextWindowSourceNative, Err: errors.Join(failures...)}
			}
		} else {
			failures = append(failures, fmt.Errorf("read model metadata: %w", err))
		}
	}

	fallback := contextWindowFallback(model, errors.Join(failures...))
	return fallback
}

func contextWindowFallback(model LaunchModel, err error) contextWindowResolution {
	return contextWindowResolution{
		ContextLength: launchModelContextLength(model),
		Source:        contextWindowSourceInventory,
		Err:           err,
	}
}

func launchModelContextLength(model LaunchModel) int {
	if model.ContextLength > 0 {
		return model.ContextLength
	}
	if model.Details.ContextLength > 0 {
		return model.Details.ContextLength
	}
	return 0
}

func modelParameterNumCtx(parameters string) (int, bool) {
	scanner := bufio.NewScanner(strings.NewReader(parameters))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "num_ctx" {
			continue
		}
		n, err := strconv.Atoi(fields[1])
		if err == nil && n > 0 {
			return n, true
		}
	}
	return 0, false
}

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

package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

type preloadModelSpec struct {
	Name      string
	Prompt    string
	KeepAlive *api.Duration
	Options   map[string]any
	Think     *api.ThinkValue
}

func parsePreloadSpecs(raw string) ([]preloadModelSpec, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}

	parts := strings.Split(raw, ",")
	specs := make([]preloadModelSpec, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		spec, err := parsePreloadEntry(part)
		if err != nil {
			return nil, fmt.Errorf("could not parse %q: %w", part, err)
		}

		specs = append(specs, spec)
	}

	return specs, nil
}

func parsePreloadEntry(raw string) (preloadModelSpec, error) {
	var spec preloadModelSpec

	namePart, optionsPart, hasOptions := strings.Cut(raw, "?")
	spec.Name = strings.TrimSpace(namePart)
	if spec.Name == "" {
		return spec, fmt.Errorf("model name is required")
	}

	if !hasOptions || strings.TrimSpace(optionsPart) == "" {
		return spec, nil
	}

	values, err := url.ParseQuery(optionsPart)
	if err != nil {
		return spec, fmt.Errorf("invalid parameters: %w", err)
	}

	opts := map[string]any{}
	for key, val := range values {
		if len(val) == 0 {
			continue
		}

		v := val
		// only use the last value unless multiple are explicitly provided
		if len(v) == 1 {
			switch strings.ToLower(key) {
			case "prompt":
				spec.Prompt = v[0]
				continue
			case "keepalive", "keep_alive":
				d, err := parseDurationValue(v[0])
				if err != nil {
					return spec, fmt.Errorf("keepalive for %s: %w", spec.Name, err)
				}
				spec.KeepAlive = &api.Duration{Duration: d}
				continue
			case "think":
				tv, err := parseThinkValue(v[0])
				if err != nil {
					return spec, fmt.Errorf("think for %s: %w", spec.Name, err)
				}
				spec.Think = tv
				continue
			}

			opts[key] = parseValue(v[0])
			continue
		}

		values := make([]any, 0, len(v))
		for _, vv := range v {
			values = append(values, parseValue(vv))
		}
		opts[key] = values
	}

	if len(opts) > 0 {
		spec.Options = opts
	}

	return spec, nil
}

func parseDurationValue(raw string) (time.Duration, error) {
	if d, err := time.ParseDuration(raw); err == nil {
		return d, nil
	}

	seconds, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid duration %q", raw)
	}

	return time.Duration(seconds) * time.Second, nil
}

func parseThinkValue(raw string) (*api.ThinkValue, error) {
	lowered := strings.ToLower(raw)
	switch lowered {
	case "true", "false":
		b, _ := strconv.ParseBool(lowered)
		return &api.ThinkValue{Value: b}, nil
	case "high", "medium", "low":
		return &api.ThinkValue{Value: lowered}, nil
	default:
		return nil, fmt.Errorf("invalid think value %q", raw)
	}
}

func parseValue(raw string) any {
	if b, err := strconv.ParseBool(raw); err == nil {
		return b
	}

	if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return i
	}

	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		return f
	}

	return raw
}

func waitForServerReady(ctx context.Context, client *api.Client) error {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		if err := client.Heartbeat(ctx); err == nil {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func preloadModels(ctx context.Context) error {
	specs, err := parsePreloadSpecs(envconfig.PreloadedModels())
	if err != nil {
		return err
	}

	if len(specs) == 0 {
		return nil
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if err := waitForServerReady(ctx, client); err != nil {
		return err
	}

	for _, spec := range specs {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		slog.Info("preloading model", "model", spec.Name)

		info, err := client.Show(ctx, &api.ShowRequest{Name: spec.Name})
		if err != nil {
			slog.Error("unable to describe model for preloading", "model", spec.Name, "error", err)
			continue
		}

		isEmbedding := slices.Contains(info.Capabilities, model.CapabilityEmbedding)
		prompt := spec.Prompt
		if prompt == "" && isEmbedding {
			prompt = "init"
		}

		if spec.Options == nil {
			spec.Options = map[string]any{}
		}

		if isEmbedding {
			req := &api.EmbedRequest{
				Model:     spec.Name,
				Input:     prompt,
				KeepAlive: spec.KeepAlive,
				Options:   spec.Options,
			}

			if _, err := client.Embed(ctx, req); err != nil {
				slog.Error("preloading embedding model failed", "model", spec.Name, "error", err)
				continue
			}
		} else {
			stream := false
			req := &api.GenerateRequest{
				Model:     spec.Name,
				Prompt:    prompt,
				KeepAlive: spec.KeepAlive,
				Options:   spec.Options,
				Stream:    &stream,
				Think:     spec.Think,
			}

			if err := client.Generate(ctx, req, func(api.GenerateResponse) error { return nil }); err != nil {
				slog.Error("preloading model failed", "model", spec.Name, "error", err)
				continue
			}
		}

		slog.Info("model preloaded", "model", spec.Name)
	}

	return nil
}

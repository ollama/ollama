package server

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) thinkingInputError(ctx context.Context, name string, inputErr error) error {
	ref, err := parseAndValidateModelRef(name)
	if err != nil {
		return inputErr
	}
	var thinking *model.Thinking
	if ref.Source == modelSourceCloud {
		if disabled, _ := cloud.Status(); disabled {
			return inputErr
		}
		cache := newModelShowCache()
		if s.modelCaches != nil && s.modelCaches.show != nil {
			cache = s.modelCaches.show
		}
		key := modelShowCloudKeyForModel(ref.Base, false)
		info, ok := cache.getCloud(key)
		if !ok {
			info, err = cache.fetchCloudShow(ctx, ref.Base, false)
			if err != nil {
				return inputErr
			}
			cache.setCloud(key, info)
		}
		thinking = info.Thinking
	} else if name, err := getExistingName(ref.Name); err == nil {
		if m, err := GetModel(name.String()); err == nil {
			thinking = m.Thinking()
		}
	}
	if !thinking.Valid() {
		return inputErr
	}
	values, _ := json.Marshal(thinking.Values)
	return fmt.Errorf("%w; supported values: %s", inputErr, values)
}

// Exact template identities keep legacy metadata from leaking to custom templates.
// These entries describe the existing local endpoint behavior, without migrations.
var legacyThinking = map[string]model.Thinking{
	// Qwen3 template with explicit /think and /no_think controls.
	"sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4": {Values: []any{false, true}, Default: true},
	// Older Qwen3 template always opens a thinking block.
	"sha256:2d54db2b9bb29ce7db54fea63a891f5859603813c555b1f88b5e0994652897f9": {Values: []any{true}, Default: true},
	// Llama3.2 tool-use template.
	"sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396": {Values: []any{false}, Default: false},
}

// Thinking returns the effective local serving contract. Remote models obtain
// their current contract from the remote show response.
func (m *Model) Thinking() *model.Thinking {
	if m == nil || m.Config.RemoteHost != "" {
		return nil
	}
	if shouldUseHarmony(m) {
		return renderers.ThinkingForRenderer("harmony")
	}
	if name := resolveRendererName(m); name != "" {
		thinking := renderers.ThinkingForRenderer(name)
		if thinking == nil {
			return nil
		}
		// Discovery must match the controls accepted by the serving boundary.
		if !slices.Contains(m.Capabilities(), model.CapabilityThinking) {
			return &model.Thinking{Values: []any{false}, Default: false}
		}
		// Preserve the local endpoint's historical default-on behavior.
		if thinking.Default == false && thinking.Supports(true) {
			thinking.Default = true
		}
		// true currently reaches Qwen3.8 as medium via ThinkValue.String().
		if name == "qwen3.8" {
			thinking.Default = "medium"
		}
		return thinking
	}
	if shouldUseGoTemplate(m) {
		if thinking, ok := legacyThinking[m.templateDigest]; ok {
			return thinking.Clone()
		}
	}
	return nil
}

// genericThinking excludes Harmony and template-only paths from new fallback rules.
func (m *Model) genericThinking() *model.Thinking {
	if m == nil || m.Config.Renderer == "" || shouldUseHarmony(m) {
		return nil
	}
	return m.Thinking()
}

// lookupThinking lets compatibility middleware preserve named efforts only for
// local generic renderers. Other models keep the existing protocol conversions.
func lookupThinking(name string) *model.Thinking {
	ref, err := parseAndValidateModelRef(name)
	if err != nil || ref.Source == modelSourceCloud {
		return nil
	}
	canonical, err := getExistingName(ref.Name)
	if err != nil {
		return nil
	}
	m, err := GetModel(canonical.String())
	if err != nil {
		return nil
	}
	return m.genericThinking()
}

// Cloud Responses normally pass through unchanged. Local web-search conversion
// needs the remote controls so it does not replace model-defined effort names
// with legacy aliases or lose boolean thinking on default-off models.
func (s *Server) responsesMiddleware(c *gin.Context) {
	middleware.ResponsesMiddleware(func(name string) *model.Thinking {
		ref, err := parseAndValidateModelRef(name)
		if err != nil || ref.Source != modelSourceCloud {
			return lookupThinking(name)
		}
		if disabled, _ := cloud.Status(); disabled {
			return nil
		}
		cache := newModelShowCache()
		if s.modelCaches != nil && s.modelCaches.show != nil {
			cache = s.modelCaches.show
		}
		ctx := c.Request.Context()
		info, ok := cache.GetCloudSWR(ctx, api.ShowRequest{Model: ref.Base})
		if !ok {
			info, err = cache.fetchCloudShow(ctx, ref.Base, false)
			if err != nil {
				return nil
			}
			cache.setCloud(modelShowCloudKeyForModel(ref.Base, false), info)
		}
		// Cloud Harmony, like local Harmony, retains legacy effort aliases.
		if info.Details.Family == "gptoss" {
			return nil
		}
		return info.Thinking
	})(c)
}

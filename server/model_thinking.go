package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/openai"
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
		info, err := cache.cloudThinking(c.Request.Context(), ref.Base)
		if err != nil {
			status := http.StatusServiceUnavailable
			code := "thinking_metadata_unavailable"
			message := "Cloud thinking controls are temporarily unavailable; retry this request."
			var authErr api.AuthorizationError
			var statusErr api.StatusError
			if errors.As(err, &authErr) {
				signinURL := authErr.SigninURL
				if signinURL == "" {
					if fallback, err := cloudProxySigninURL(); err == nil {
						signinURL = fallback
					}
				}
				c.AbortWithStatusJSON(http.StatusUnauthorized, openai.NewResponsesAuthenticationError(signinURL))
				return nil
			}
			if errors.As(err, &statusErr) && (statusErr.StatusCode == http.StatusForbidden || statusErr.StatusCode == http.StatusNotFound || statusErr.StatusCode == http.StatusTooManyRequests) {
				status, code, message = statusErr.StatusCode, "cloud_model_unavailable", "Cannot read this cloud model's thinking controls; check model availability and cloud access."
			}
			if status == http.StatusServiceUnavailable || status == http.StatusTooManyRequests {
				c.Header("Retry-After", "1")
			}
			response := openai.NewError(status, message)
			response.Error.Code = &code
			c.AbortWithStatusJSON(status, response)
			return nil
		}
		return info.Thinking
	})(c)
}

const cloudThinkingWait = 500 * time.Millisecond

func (c *modelShowCache) cloudThinking(ctx context.Context, name string) (*api.ShowResponse, error) {
	usable := func(info *api.ShowResponse) bool {
		return info != nil && info.Thinking.Valid()
	}
	if info, ok := c.GetCloudSWR(ctx, api.ShowRequest{Model: name}); ok {
		if !usable(info) {
			return nil, errors.New("cloud thinking metadata is missing or invalid")
		}
		return info, nil
	}
	key := modelShowCloudKeyForModel(name, false)
	result := c.cloudThinkingLoads.DoChan(key.Model, func() (any, error) {
		if info, ok := c.getCloud(key); ok && usable(info) {
			return info, nil
		}
		// A caller's short wait must not cancel the fetch that can serve its retry.
		// fetchCloudShow bounds the background request with the cache's own timeout.
		info, err := c.fetchCloudShow(context.WithoutCancel(ctx), key.Model, false)
		if err != nil {
			return nil, err
		}
		if !usable(info) {
			return nil, errors.New("cloud thinking metadata is missing or invalid")
		}
		c.setCloud(key, info)
		c.mu.Lock()
		c.cloudNextReadRefreshAfter[key] = time.Now().Add(modelShowCloudReadRefreshCooldown)
		c.mu.Unlock()
		return info, nil
	})
	timer := time.NewTimer(cloudThinkingWait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
		return nil, context.DeadlineExceeded
	case value := <-result:
		if value.Err != nil {
			return nil, value.Err
		}
		return value.Val.(*api.ShowResponse), nil
	}
}

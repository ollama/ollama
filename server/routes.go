package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/registry"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
	"github.com/ollama/ollama/x/imagegen"
	xserver "github.com/ollama/ollama/x/server"
)

const signinURLStr = "https://ollama.com/connect?name=%s&key=%s"

func shouldUseHarmony(model *Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, model.Config.ModelFamily) {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

func experimentEnabled(name string) bool {
	return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")

// Low VRAM mode is based on the sum of total VRAM (not free) and triggers
// reduced context length on some models
var lowVRAMThreshold uint64 = 20 * format.GibiByte

var mode string = gin.DebugMode

type Server struct {
	addr    net.Addr
	sched   *Scheduler
	lowVRAM bool
}

func init() {
	switch mode {
	case gin.DebugMode:
	case gin.ReleaseMode:
	case gin.TestMode:
	default:
		mode = gin.DebugMode
	}

	gin.SetMode(mode)

	// Tell renderers to use [img] tags
	renderers.RenderImgTags = true
}

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

func modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	opts := api.DefaultOptions()
	if err := opts.FromMap(model.Options); err != nil {
		return api.Options{}, err
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	return opts, nil
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, and consolidated options if successful and error otherwise.
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	model, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
	}

	if slices.Contains(model.Config.ModelFamilies, "mllama") && len(model.ProjectorPaths) > 0 {
		return nil, nil, nil, fmt.Errorf("'llama3.2-vision' is no longer compatible with your version of Ollama and has been replaced by a newer version. To re-download, run 'ollama pull llama3.2-vision'")
	}

	if err := model.CheckCapabilities(caps...); err != nil {
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	opts, err := modelOptions(model, requestOpts)
	if err != nil {
		return nil, nil, nil, err
	}

	// This model is much more capable with a larger context, so set that
	// unless it would penalize performance too much
	if !s.lowVRAM && slices.Contains([]string{
		"gptoss", "gpt-oss",
		"qwen3vl", "qwen3vlmoe",
	}, model.Config.ModelFamily) {
		opts.NumCtx = max(opts.NumCtx, 8192)
	}

	runnerCh, errCh := s.sched.GetRunner(ctx, model, opts, keepAlive)
	var runner *runnerRef
	select {
	case runner = <-runnerCh:
	case err = <-errCh:
		return nil, nil, nil, err
	}

	return runner.llama, model, &opts, nil
}

func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf(signinURLStr, url.PathEscape(h), encKey), nil
}

func (s *Server) GenerateHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		// Ideally this is "invalid model name" but we're keeping with
		// what the API currently returns until we can change it.
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// We cannot currently consolidate this into GetModel because all we'll
	// induce infinite recursion given the current code structure.
	name, err := getExistingName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	// Handle image generation models
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		s.handleImageGenerate(c, req, name.String(), checkpointStart)
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		origModel := req.Model

		remoteURL, err := url.Parse(m.Config.RemoteHost)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if !slices.Contains(envconfig.Remotes(), remoteURL.Hostname()) {
			slog.Info("remote model", "remotes", envconfig.Remotes(), "remoteURL", m.Config.RemoteHost, "hostname", remoteURL.Hostname())
			c.JSON(http.StatusBadRequest, gin.H{"error": "this server cannot run this remote model"})
			return
		}

		req.Model = m.Config.RemoteModel

		if req.Template == "" && m.Template.String() != "" {
			req.Template = m.Template.String()
		}

		if req.Options == nil {
			req.Options = map[string]any{}
		}

		for k, v := range m.Options {
			if _, ok := req.Options[k]; !ok {
				req.Options[k] = v
			}
		}

		// update the system prompt from the model if one isn't already specified
		if req.System == "" && m.System != "" {
			req.System = m.System
		}

		if len(m.Messages) > 0 {
			slog.Warn("embedded messages in the model not supported with '/api/generate'; try '/api/chat' instead")
		}

		contentType := "application/x-ndjson"
		if req.Stream != nil && !*req.Stream {
			contentType = "application/json; charset=utf-8"
		}
		c.Header("Content-Type", contentType)

		fn := func(resp api.GenerateResponse) error {
			resp.Model = origModel
			resp.RemoteModel = m.Config.RemoteModel
			resp.RemoteHost = m.Config.RemoteHost

			data, err := json.Marshal(resp)
			if err != nil {
				return err
			}

			if _, err = c.Writer.Write(append(data, '\n')); err != nil {
				return err
			}
			c.Writer.Flush()
			return nil
		}

		client := api.NewClient(remoteURL, http.DefaultClient)
		err = client.Generate(c, &req, fn)
		if err != nil {
			var authError api.AuthorizationError
			if errors.As(err, &authError) {
				sURL, sErr := signinURL()
				if sErr != nil {
					slog.Error(sErr.Error())
					c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
					return
				}

				c.JSON(authError.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
				return
			}
			var apiError api.StatusError
			if errors.As(err, &apiError) {
				c.JSON(apiError.StatusCode, apiError)
				return
			}
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		return
	}

	// expire the runner
	if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	var builtinParser parsers.Parser
	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	if !req.Raw && m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			// no tools or last message for generate endpoint
			builtinParser.Init(nil, nil, req.Think)
		}
	}

	// Validate Think value: string values currently only allowed for harmony/gptoss models
	if req.Think != nil && req.Think.IsString() && m.Config.Parser != "harmony" {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("think value %q is not supported for this model", req.Think.String())})
		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		caps = append(caps, model.CapabilityInsert)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	// load the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image while more than one image requested"})
		return
	}

	images := make([]llm.ImageData, len(req.Images))
	for i := range req.Images {
		images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
	}

	prompt := req.Prompt
	if !req.Raw {
		tmpl := m.Template
		if req.Template != "" {
			tmpl, err = template.Parse(req.Template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		}

		var values template.Values
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			var msgs []api.Message
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			} else if m.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: m.System})
			}

			if req.Context == nil {
				msgs = append(msgs, m.Messages...)
			}

			userMsg := api.Message{Role: "user", Content: req.Prompt}
			for _, i := range images {
				userMsg.Images = append(userMsg.Images, i.Data)
			}
			values.Messages = append(msgs, userMsg)
		}

		values.Think = req.Think != nil && req.Think.Bool()
		values.ThinkLevel = ""
		if req.Think != nil {
			values.ThinkLevel = req.Think.String()
		}
		values.IsThinkSet = req.Think != nil

		var b bytes.Buffer
		if req.Context != nil {
			slog.Warn("the context field is deprecated and will be removed in a future version of Ollama")
			s, err := r.Detokenize(c.Request.Context(), req.Context)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			b.WriteString(s)
		}

		// check that we're in the `api/chat`-like flow, and if so, generate the
		// prompt the same way
		// TEMP(drifkin): we should really just detect the chat-like flow and call
		// the real chat handler, but doing this as a stopgap to get renderer
		// support for generate
		if values.Messages != nil && values.Suffix == "" && req.Template == "" {
			prompt, images, err = chatPrompt(c.Request.Context(), m, r.Tokenize, opts, values.Messages, []api.Tool{}, req.Think, req.Truncate == nil || *req.Truncate)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			// TEMP(drifkin): req.Context will be removed very soon, but we're temporarily supporting it in this flow here
			if req.Context != nil {
				b.WriteString(prompt)
				prompt = b.String()
			}
		} else {
			// legacy flow
			if err := tmpl.Execute(&b, values); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			prompt = b.String()
		}
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(images),
			},
		})
		return
	}

	var thinkingState *thinking.Parser
	if builtinParser == nil {
		openingTag, closingTag := thinking.InferTags(m.Template.Template)
		if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
			thinkingState = &thinking.Parser{
				OpeningTag: openingTag,
				ClosingTag: closingTag,
			}
			if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
				thinkingState.AddContent(openingTag)
			}
		}
	}

	ch := make(chan any)
	go func() {
		// TODO (jmorganca): avoid building the response twice both here and below
		var sb strings.Builder
		defer close(ch)
		if err := r.Completion(c.Request.Context(), llm.CompletionRequest{
			Prompt:      prompt,
			Images:      images,
			Format:      req.Format,
			Options:     opts,
			Shift:       req.Shift == nil || *req.Shift,
			Truncate:    req.Truncate == nil || *req.Truncate,
			Logprobs:    req.Logprobs,
			TopLogprobs: req.TopLogprobs,
		}, func(cr llm.CompletionResponse) {
			res := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Response:  cr.Content,
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
				Logprobs: toAPILogprobs(cr.Logprobs),
			}

			if builtinParser != nil {
				content, thinking, toolCalls, err := builtinParser.Add(cr.Content, cr.Done)
				if err != nil {
					ch <- gin.H{"error": err.Error()}
					return
				}
				res.Response = content
				res.Thinking = thinking
				if cr.Done && len(toolCalls) > 0 {
					res.ToolCalls = toolCalls
				}
			} else if thinkingState != nil {
				thinking, content := thinkingState.AddContent(cr.Content)
				res.Thinking = thinking
				res.Response = content
			}

			if _, err := sb.WriteString(cr.Content); err != nil {
				ch <- gin.H{"error": err.Error()}
			}

			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

				if !req.Raw {
					tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
					res.Context = tokens
				}
			}

			if builtinParser != nil {
				// only send messages with meaningful content (empty messages confuse clients)
				if res.Response != "" || res.Thinking != "" || res.Done || len(res.ToolCalls) > 0 {
					ch <- res
				}

				return
			}

			ch <- res
		}); err != nil {
			var serr api.StatusError
			if errors.As(err, &serr) {
				ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
			} else {
				ch <- gin.H{"error": err.Error()}
			}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var r api.GenerateResponse
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.GenerateResponse:
				sbThinking.WriteString(t.Thinking)
				sbContent.WriteString(t.Response)
				r = t
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		r.Thinking = sbThinking.String()
		r.Response = sbContent.String()
		r.Logprobs = allLogprobs

		c.JSON(http.StatusOK, r)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}

	name, err := getExistingName(model.ParseName(req.Model))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()

	embedWithRetry := func(text string) ([]float32, int, error) {
		emb, tokCount, err := r.Embedding(ctx, text)
		if err == nil {
			return emb, tokCount, nil
		}

		var serr api.StatusError
		if !errors.As(err, &serr) || serr.StatusCode != http.StatusBadRequest {
			return nil, 0, err
		}
		if req.Truncate != nil && !*req.Truncate {
			return nil, 0, err
		}

		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return nil, 0, err
		}

		// TODO @nicolepardal: avoid reaching into kvData here; pass required tokenizer metadata via model/options instead
		ctxLen := min(opts.NumCtx, int(kvData.ContextLength()))
		if bos := kvData.Uint("tokenizer.ggml.bos_token_id"); len(tokens) > 0 && tokens[0] != int(bos) && kvData.Bool("add_bos_token", true) {
			ctxLen--
		}
		if eos := kvData.Uint("tokenizer.ggml.eos_token_id"); len(tokens) > 0 && tokens[len(tokens)-1] != int(eos) && kvData.Bool("add_eos_token", true) {
			ctxLen--
		}

		if len(tokens) <= ctxLen {
			return nil, 0, fmt.Errorf("input exceeds maximum context length and cannot be truncated further")
		}
		if ctxLen <= 0 {
			return nil, 0, fmt.Errorf("input after truncation exceeds maximum context length")
		}

		truncatedTokens := tokens[:ctxLen]
		truncated, err := r.Detokenize(ctx, truncatedTokens)
		if err != nil {
			return nil, 0, err
		}
		return r.Embedding(ctx, truncated)
	}

	var g errgroup.Group
	embeddings := make([][]float32, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			embedding, tokenCount, err := embedWithRetry(text)
			if err != nil {
				return err
			}
			// TODO: this first normalization should be done by the model
			embedding, err = normalize(embedding)
			if err != nil {
				return err
			}
			if req.Dimensions > 0 && req.Dimensions < len(embedding) {
				embedding, err = normalize(embedding[:req.Dimensions])
				if err != nil {
					return err
				}
			}
			embeddings[i] = embedding
			atomic.AddUint64(&totalTokens, uint64(tokenCount))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{
				"error": strings.TrimSpace(serr.ErrorMessage),
			})
			return
		}

		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": strings.TrimSpace(err.Error()),
		})
		return
	}

	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	}
	c.JSON(http.StatusOK, resp)
}

func normalize(vec []float32) ([]float32, error) {
	var sum float32
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("embedding contains NaN or Inf values")
		}
		sum += v * v
	}

	norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
	for i := range vec {
		vec[i] *= norm
	}
	return vec, nil
}

func (s *Server) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	embedding, _, err := r.Embedding(c.Request.Context(), req.Prompt)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := api.EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) PullHandler(c *gin.Context) {
	var req api.PullRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name := model.ParseName(cmp.Or(req.Model, req.Name))
	if !name.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
		return
	}

	name, err = getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		if err := PullModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) PushHandler(c *gin.Context) {
	var req api.PushRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var mname string
	if req.Model != "" {
		mname = req.Model
	} else if req.Name != "" {
		mname = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		name, err := getExistingName(model.ParseName(mname))
		if err != nil {
			ch <- gin.H{"error": err.Error()}
			return
		}

		if err := PushModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

// getExistingName searches the models directory for the longest prefix match of
// the input name and returns the input name with all existing parts replaced
// with each part found. If no parts are found, the input name is returned as
// is.
func getExistingName(n model.Name) (model.Name, error) {
	var zero model.Name
	existing, err := Manifests(true)
	if err != nil {
		return zero, err
	}
	var set model.Name // tracks parts already canonicalized
	for e := range existing {
		if set.Host == "" && strings.EqualFold(e.Host, n.Host) {
			n.Host = e.Host
		}
		if set.Namespace == "" && strings.EqualFold(e.Namespace, n.Namespace) {
			n.Namespace = e.Namespace
		}
		if set.Model == "" && strings.EqualFold(e.Model, n.Model) {
			n.Model = e.Model
		}
		if set.Tag == "" && strings.EqualFold(e.Tag, n.Tag) {
			n.Tag = e.Tag
		}
	}
	return n, nil
}

func (s *Server) DeleteHandler(c *gin.Context) {
	var r api.DeleteRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	n := model.ParseName(cmp.Or(r.Model, r.Name))
	if !n.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("name %q is invalid", cmp.Or(r.Model, r.Name))})
		return
	}

	n, err := getExistingName(n)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		return
	}

	m, err := ParseNamedManifest(n)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if err := m.Remove(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if err := m.RemoveLayers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func (s *Server) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	resp, err := GetModelInfo(req)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	c.JSON(http.StatusOK, resp)
}

func GetModelInfo(req api.ShowRequest) (*api.ShowResponse, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, ErrModelPathInvalid
	}
	name, err := getExistingName(name)
	if err != nil {
		return nil, err
	}

	m, err := GetModel(name.String())
	if err != nil {
		return nil, err
	}

	modelDetails := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.Config.ModelFamily,
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}

	// For image generation models, populate details from imagegen package
	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		if info, err := imagegen.GetModelInfo(name.String()); err == nil {
			modelDetails.Family = info.Architecture
			modelDetails.ParameterSize = format.HumanNumber(uint64(info.ParameterCount))
			modelDetails.QuantizationLevel = info.Quantization
		}
	}

	// For safetensors LLM models (experimental), populate details from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name.String()); err == nil {
			if arch, ok := info["general.architecture"].(string); ok && arch != "" {
				modelDetails.Family = arch
			}
			if paramCount, ok := info["general.parameter_count"].(int64); ok && paramCount > 0 {
				modelDetails.ParameterSize = format.HumanNumber(uint64(paramCount))
			}
		}
		// Get torch_dtype directly from config.json for quantization level
		if dtype, err := xserver.GetSafetensorsDtype(name.String()); err == nil && dtype != "" {
			modelDetails.QuantizationLevel = dtype
		}
	}

	if req.System != "" {
		m.System = req.System
	}

	msgs := make([]api.Message, len(m.Messages))
	for i, msg := range m.Messages {
		msgs[i] = api.Message{Role: msg.Role, Content: msg.Content}
	}

	manifest, err := ParseNamedManifest(name)
	if err != nil {
		return nil, err
	}

	resp := &api.ShowResponse{
		License:      strings.Join(m.License, "\n"),
		System:       m.System,
		Template:     m.Template.String(),
		Details:      modelDetails,
		Messages:     msgs,
		Capabilities: m.Capabilities(),
		ModifiedAt:   manifest.fi.ModTime(),
		Requires:     m.Config.Requires,
	}

	if m.Config.RemoteHost != "" {
		resp.RemoteHost = m.Config.RemoteHost
		resp.RemoteModel = m.Config.RemoteModel

		if m.Config.ModelFamily != "" {
			resp.ModelInfo = make(map[string]any)
			resp.ModelInfo["general.architecture"] = m.Config.ModelFamily

			if m.Config.BaseName != "" {
				resp.ModelInfo["general.basename"] = m.Config.BaseName
			}

			if m.Config.ContextLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.context_length", m.Config.ModelFamily)] = m.Config.ContextLen
			}

			if m.Config.EmbedLen > 0 {
				resp.ModelInfo[fmt.Sprintf("%s.embedding_length", m.Config.ModelFamily)] = m.Config.EmbedLen
			}
		}
	}

	var params []string
	cs := 30
	for k, v := range m.Options {
		switch val := v.(type) {
		case []any:
			for _, nv := range val {
				params = append(params, fmt.Sprintf("%-*s %#v", cs, k, nv))
			}
		default:
			params = append(params, fmt.Sprintf("%-*s %#v", cs, k, v))
		}
	}
	resp.Parameters = strings.Join(params, "\n")

	if len(req.Options) > 0 {
		if m.Options == nil {
			m.Options = make(map[string]any)
		}
		for k, v := range req.Options {
			m.Options[k] = v
		}
	}

	var sb strings.Builder
	fmt.Fprintln(&sb, "# Modelfile generated by \"ollama show\"")
	fmt.Fprintln(&sb, "# To build a new Modelfile based on this, replace FROM with:")
	fmt.Fprintf(&sb, "# FROM %s\n\n", m.ShortName)
	fmt.Fprint(&sb, m.String())
	resp.Modelfile = sb.String()

	// skip loading tensor information if this is a remote model
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		return resp, nil
	}

	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name.String()); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	// For safetensors LLM models (experimental), populate ModelInfo from config.json
	if m.Config.ModelFormat == "safetensors" && slices.Contains(m.Config.Capabilities, "completion") {
		if info, err := xserver.GetSafetensorsLLMInfo(name.String()); err == nil {
			resp.ModelInfo = info
		}
		// Populate tensor info if verbose
		if req.Verbose {
			if tensors, err := xserver.GetSafetensorsTensorInfo(name.String()); err == nil {
				resp.Tensors = tensors
			}
		}
		return resp, nil
	}

	kvData, tensors, err := getModelData(m.ModelPath, req.Verbose)
	if err != nil {
		return nil, err
	}

	delete(kvData, "general.name")
	delete(kvData, "tokenizer.chat_template")
	resp.ModelInfo = kvData

	tensorData := make([]api.Tensor, len(tensors.Items()))
	for cnt, t := range tensors.Items() {
		tensorData[cnt] = api.Tensor{Name: t.Name, Type: t.Type(), Shape: t.Shape}
	}
	resp.Tensors = tensorData

	if len(m.ProjectorPaths) > 0 {
		projectorData, _, err := getModelData(m.ProjectorPaths[0], req.Verbose)
		if err != nil {
			return nil, err
		}
		resp.ProjectorInfo = projectorData
	}

	return resp, nil
}

func getModelData(digest string, verbose bool) (ggml.KV, ggml.Tensors, error) {
	maxArraySize := 0
	if verbose {
		maxArraySize = -1
	}
	data, err := llm.LoadModel(digest, maxArraySize)
	if err != nil {
		return nil, ggml.Tensors{}, err
	}

	kv := data.KV()

	if !verbose {
		for k := range kv {
			if t, ok := kv[k].([]any); len(t) > 5 && ok {
				kv[k] = []any{}
			}
		}
	}

	return kv, data.Tensors(), nil
}

func (s *Server) ListHandler(c *gin.Context) {
	ms, err := Manifests(true)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	models := []api.ListModelResponse{}
	for n, m := range ms {
		var cf model.ConfigV2

		if m.Config.Digest != "" {
			f, err := m.Config.Open()
			if err != nil {
				slog.Warn("bad manifest filepath", "name", n, "error", err)
				continue
			}
			defer f.Close()

			if err := json.NewDecoder(f).Decode(&cf); err != nil {
				slog.Warn("bad manifest config", "name", n, "error", err)
				continue
			}
		}

		// tag should never be masked
		models = append(models, api.ListModelResponse{
			Model:       n.DisplayShortest(),
			Name:        n.DisplayShortest(),
			RemoteModel: cf.RemoteModel,
			RemoteHost:  cf.RemoteHost,
			Size:        m.Size(),
			Digest:      m.digest,
			ModifiedAt:  m.fi.ModTime(),
			Details: api.ModelDetails{
				Format:            cf.ModelFormat,
				Family:            cf.ModelFamily,
				Families:          cf.ModelFamilies,
				ParameterSize:     cf.ModelType,
				QuantizationLevel: cf.FileType,
			},
		})
	}

	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// most recently modified first
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})

	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func (s *Server) CopyHandler(c *gin.Context) {
	var r api.CopyRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src := model.ParseName(r.Source)
	if !src.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("source %q is invalid", r.Source)})
		return
	}
	src, err := getExistingName(src)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	dst := model.ParseName(r.Destination)
	if !dst.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("destination %q is invalid", r.Destination)})
		return
	}
	dst, err = getExistingName(dst)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := CopyModel(src, dst); errors.Is(err, os.ErrNotExist) {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", r.Source)})
	} else if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func (s *Server) HeadBlobHandler(c *gin.Context) {
	path, err := GetBlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if _, err := os.Stat(path); err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("blob %q not found", c.Param("digest"))})
		return
	}

	c.Status(http.StatusOK)
}

func (s *Server) CreateBlobHandler(c *gin.Context) {
	if ib, ok := intermediateBlobs[c.Param("digest")]; ok {
		p, err := GetBlobsPath(ib)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, c.Param("digest"))
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		} else {
			c.Status(http.StatusOK)
			return
		}
	}

	path, err := GetBlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	default:
		c.Status(http.StatusOK)
		return
	}

	layer, err := NewLayer(c.Request.Body, "")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if layer.Digest != c.Param("digest") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("digest mismatch, expected %q, got %q", c.Param("digest"), layer.Digest)})
		return
	}

	c.Status(http.StatusCreated)
}

func isLocalIP(ip netip.Addr) bool {
	if interfaces, err := net.Interfaces(); err == nil {
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}

			for _, a := range addrs {
				if parsed, _, err := net.ParseCIDR(a.String()); err == nil {
					if parsed.String() == ip.String() {
						return true
					}
				}
			}
		}
	}

	return false
}

func allowedHost(host string) bool {
	host = strings.ToLower(host)

	if host == "" || host == "localhost" {
		return true
	}

	if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
		return true
	}

	tlds := []string{
		"localhost",
		"local",
		"internal",
	}

	// check if the host is a local TLD
	for _, tld := range tlds {
		if strings.HasSuffix(host, "."+tld) {
			return true
		}
	}

	return false
}

func allowedHostsMiddleware(addr net.Addr) gin.HandlerFunc {
	return func(c *gin.Context) {
		if addr == nil {
			c.Next()
			return
		}

		if addr, err := netip.ParseAddrPort(addr.String()); err == nil && !addr.Addr().IsLoopback() {
			c.Next()
			return
		}

		host, _, err := net.SplitHostPort(c.Request.Host)
		if err != nil {
			host = c.Request.Host
		}

		if addr, err := netip.ParseAddr(host); err == nil {
			if addr.IsLoopback() || addr.IsPrivate() || addr.IsUnspecified() || isLocalIP(addr) {
				c.Next()
				return
			}
		}

		if allowedHost(host) {
			if c.Request.Method == http.MethodOptions {
				c.AbortWithStatus(http.StatusNoContent)
				return
			}

			c.Next()
			return
		}

		c.AbortWithStatus(http.StatusForbidden)
	}
}

func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",

		// OpenAI compatibility headers
		"OpenAI-Beta",
		"x-stainless-arch",
		"x-stainless-async",
		"x-stainless-custom-poll-interval",
		"x-stainless-helper-method",
		"x-stainless-lang",
		"x-stainless-os",
		"x-stainless-package-version",
		"x-stainless-poll-helper",
		"x-stainless-retry-count",
		"x-stainless-runtime",
		"x-stainless-runtime-version",
		"x-stainless-timeout",
	}
	corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true
	r.Use(
		cors.New(corsConfig),
		allowedHostsMiddleware(s.addr),
	)

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })

	// Local model cache management (new implementation is at end of function)
	r.POST("/api/pull", s.PullHandler)
	r.POST("/api/push", s.PushHandler)
	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.DELETE("/api/delete", s.DeleteHandler)

	r.POST("/api/me", s.WhoamiHandler)

	r.POST("/api/signout", s.SignoutHandler)
	// deprecated
	r.DELETE("/api/user/keys/:encodedKey", s.SignoutHandler)

	// Create
	r.POST("/api/create", s.CreateHandler)
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)
	r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)
	r.POST("/api/copy", s.CopyHandler)

	// Inference
	r.GET("/api/ps", s.PsHandler)
	r.POST("/api/generate", s.GenerateHandler)
	r.POST("/api/chat", s.ChatHandler)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)

	// Inference (OpenAI compatibility)
	r.POST("/v1/chat/completions", middleware.ChatMiddleware(), s.ChatHandler)
	r.POST("/v1/completions", middleware.CompletionsMiddleware(), s.GenerateHandler)
	r.POST("/v1/embeddings", middleware.EmbeddingsMiddleware(), s.EmbedHandler)
	r.GET("/v1/models", middleware.ListMiddleware(), s.ListHandler)
	r.GET("/v1/models/:model", middleware.RetrieveMiddleware(), s.ShowHandler)
	r.POST("/v1/responses", middleware.ResponsesMiddleware(), s.ChatHandler)
	// OpenAI-compatible image generation endpoint
	r.POST("/v1/images/generations", middleware.ImageGenerationsMiddleware(), s.GenerateHandler)

	// Inference (Anthropic compatibility)
	r.POST("/v1/messages", middleware.AnthropicMessagesMiddleware(), s.ChatHandler)

	if rc != nil {
		// wrap old with new
		rs := &registry.Local{
			Client:   rc,
			Logger:   slog.Default(), // TODO(bmizerany): Take a logger, do not use slog.Default()
			Fallback: r,

			Prune: PruneLayers,
		}
		return rs, nil
	}

	return r, nil
}

func Serve(ln net.Listener) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("server config", "env", envconfig.Values())

	blobsDir, err := GetBlobsPath("")
	if err != nil {
		return err
	}
	if err := fixBlobs(blobsDir); err != nil {
		return err
	}

	if !envconfig.NoPrune() {
		if _, err := Manifests(false); err != nil {
			slog.Warn("corrupt manifests detected, skipping prune operation.  Re-pull or delete to clear", "error", err)
		} else {
			// clean up unused layers and manifests
			if err := PruneLayers(); err != nil {
				return err
			}

			manifestsPath, err := GetManifestPath()
			if err != nil {
				return err
			}

			if err := PruneDirectory(manifestsPath); err != nil {
				return err
			}
		}
	}

	s := &Server{addr: ln.Addr()}

	var rc *ollama.Registry
	if useClient2 {
		var err error
		rc, err = ollama.DefaultRegistry()
		if err != nil {
			return err
		}
	}

	h, err := s.GenerateRoutes(rc)
	if err != nil {
		return err
	}

	http.Handle("/", h)

	ctx, done := context.WithCancel(context.Background())
	schedCtx, schedDone := context.WithCancel(ctx)
	sched := InitScheduler(schedCtx)
	s.sched = sched

	slog.Info(fmt.Sprintf("Listening on %s (version %s)", ln.Addr(), version.Version))
	srvr := &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		//
		// TODO(bmizerany): Decide if we want to make this
		// configurable so it is not exposed by default, or allow
		// users to bind it to a different port. This was a quick
		// and easy way to get pprof, but it may not be the best
		// way.
		Handler: nil,
	}

	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		srvr.Close()
		schedDone()
		sched.unloadAllRunners()
		done()
	}()

	s.sched.Run(schedCtx)

	// register the experimental webp decoder
	// so webp images can be used in multimodal inputs
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)

	// At startup we retrieve GPU information so we can get log messages before loading a model
	// This will log warnings to the log in case we have problems with detected GPUs
	gpus := discover.GPUDevices(ctx, nil)
	discover.LogDetails(gpus)

	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory - envconfig.GpuOverhead()
	}
	if totalVRAM < lowVRAMThreshold {
		s.lowVRAM = true
		slog.Info("entering low vram mode", "total vram", format.HumanBytes2(totalVRAM), "threshold", format.HumanBytes2(lowVRAMThreshold))
	}

	err = srvr.Serve(ln)
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	<-ctx.Done()
	return nil
}

func waitForStream(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/json")
	var latest api.ProgressResponse
	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			latest = r
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			errorMsg, ok := r["error"].(string)
			if !ok {
				errorMsg = "unknown error"
			}
			c.JSON(status, gin.H{"error": errorMsg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
			return
		}
	}

	c.JSON(http.StatusOK, latest)
}

func streamResponse(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		// errors are provided as a gin.H with an "error" field and
		// an optional "status" field.  For errors that are streamed
		// before any content, we need to set the status code and
		// content type for the error.
		if h, ok := val.(gin.H); ok {
			if e, ok := h["error"].(string); ok {
				status, ok := h["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				if !c.Writer.Written() {
					c.Header("Content-Type", "application/json")
					c.JSON(status, gin.H{"error": e})
				} else {
					if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
						slog.Error("streamResponse failed to encode json error", "error", err)
					}
				}

				return false
			}
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}

		return true
	})
}

func (s *Server) WhoamiHandler(c *gin.Context) {
	// todo allow other hosts
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	user, err := client.Whoami(c)
	if err != nil {
		slog.Error(err.Error())
	}

	// user isn't signed in
	if user != nil && user.Name == "" {
		sURL, sErr := signinURL()
		if sErr != nil {
			slog.Error(sErr.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
			return
		}

		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": sURL})
		return
	}

	c.JSON(http.StatusOK, user)
}

func (s *Server) SignoutHandler(c *gin.Context) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		slog.Error("couldn't get public key", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))

	// todo allow other hosts
	u, err := url.Parse("https://ollama.com")
	if err != nil {
		slog.Error(err.Error())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "URL parse error"})
		return
	}

	client := api.NewClient(u, http.DefaultClient)
	err = client.Disconnect(c, encKey)
	if err != nil {
		var authError api.AuthorizationError
		if errors.As(err, &authError) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "you are not currently signed in"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "there was an error signing out"})
		return
	}

	c.JSON(http.StatusOK, nil)
}

func (s *Server) PsHandler(c *gin.Context) {
	models := []api.ProcessModelResponse{}

	for _, v := range s.sched.loaded {
		model := v.model
		modelDetails := api.ModelDetails{
			Format:            model.Config.ModelFormat,
			Family:            model.Config.ModelFamily,
			Families:          model.Config.ModelFamilies,
			ParameterSize:     model.Config.ModelType,
			QuantizationLevel: model.Config.FileType,
		}

		mr := api.ProcessModelResponse{
			Model:     model.ShortName,
			Name:      model.ShortName,
			Size:      int64(v.totalSize),
			SizeVRAM:  int64(v.vramSize),
			Digest:    model.Digest,
			Details:   modelDetails,
			ExpiresAt: v.expiresAt,
		}
		if v.Options != nil {
			mr.ContextLength = v.Options.NumCtx
		}
		// The scheduler waits to set expiresAt, so if a model is loading it's
		// possible that it will be set to the unix epoch. For those cases, just
		// calculate the time w/ the sessionDuration instead.
		var epoch time.Time
		if v.expiresAt == epoch {
			mr.ExpiresAt = time.Now().Add(v.sessionDuration)
		}

		models = append(models, mr)
	}

	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		// longest duration remaining listed first
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})

	c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}

func toolCallId() string {
	const letterBytes = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return "call_" + strings.ToLower(string(b))
}

func (s *Server) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	name := model.ParseName(req.Model)
	if !name.IsValid() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	name, err := getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(req.Model)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		origModel := req.Model

		remoteURL, err := url.Parse(m.Config.RemoteHost)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if !slices.Contains(envconfig.Remotes(), remoteURL.Hostname()) {
			slog.Info("remote model", "remotes", envconfig.Remotes(), "remoteURL", m.Config.RemoteHost, "hostname", remoteURL.Hostname())
			c.JSON(http.StatusBadRequest, gin.H{"error": "this server cannot run this remote model"})
			return
		}

		req.Model = m.Config.RemoteModel
		if req.Options == nil {
			req.Options = map[string]any{}
		}

		var msgs []api.Message
		if len(req.Messages) > 0 {
			msgs = append(m.Messages, req.Messages...)
			if req.Messages[0].Role != "system" && m.System != "" {
				msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
			}
		}

		msgs = filterThinkTags(msgs, m)
		req.Messages = msgs

		for k, v := range m.Options {
			if _, ok := req.Options[k]; !ok {
				req.Options[k] = v
			}
		}

		contentType := "application/x-ndjson"
		if req.Stream != nil && !*req.Stream {
			contentType = "application/json; charset=utf-8"
		}
		c.Header("Content-Type", contentType)

		fn := func(resp api.ChatResponse) error {
			resp.Model = origModel
			resp.RemoteModel = m.Config.RemoteModel
			resp.RemoteHost = m.Config.RemoteHost

			data, err := json.Marshal(resp)
			if err != nil {
				return err
			}

			if _, err = c.Writer.Write(append(data, '\n')); err != nil {
				return err
			}
			c.Writer.Flush()
			return nil
		}

		client := api.NewClient(remoteURL, http.DefaultClient)
		err = client.Chat(c, &req, fn)
		if err != nil {
			var authError api.AuthorizationError
			if errors.As(err, &authError) {
				sURL, sErr := signinURL()
				if sErr != nil {
					slog.Error(sErr.Error())
					c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
					return
				}

				c.JSON(authError.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
				return
			}
			var apiError api.StatusError
			if errors.As(err, &apiError) {
				c.JSON(apiError.StatusCode, apiError)
				return
			}
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		return
	}

	caps := []model.Capability{model.CapabilityCompletion}
	if len(req.Tools) > 0 {
		caps = append(caps, model.CapabilityTools)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		caps = append(caps, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			// Set think to nil when being used with Anthropic API to connect to tools like claude code
			if _, ok := c.Get("relax_thinking"); ok {
				slog.Warn("model does not support thinking, relaxing thinking to nil", "model", req.Model)
				req.Think = nil
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
				return
			}
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support chat", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	msgs := append(m.Messages, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)

	if shouldUseHarmony(m) && m.Config.Parser == "" {
		m.Config.Parser = "harmony"
	}

	var builtinParser parsers.Parser
	processedTools := req.Tools

	if m.Config.Parser != "" {
		builtinParser = parsers.ParserForName(m.Config.Parser)
		if builtinParser != nil {
			// Determine last message for chat prefill
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			// Initialize parser and get processed tools
			processedTools = builtinParser.Init(req.Tools, lastMessage, req.Think)
		}
	}

	truncate := req.Truncate == nil || *req.Truncate
	prompt, images, err := chatPrompt(c.Request.Context(), m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate)
	if err != nil {
		slog.Error("chat prompt error", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(images),
			},
		})
		return
	}

	// Validate Think value: string values currently only allowed for harmony/gptoss models
	if req.Think != nil && req.Think.IsString() && m.Config.Parser != "harmony" {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("think value %q is not supported for this model", req.Think.String())})
		return
	}

	var thinkingState *thinking.Parser
	openingTag, closingTag := thinking.InferTags(m.Template.Template)
	if req.Think != nil && req.Think.Bool() && openingTag != "" && closingTag != "" {
		thinkingState = &thinking.Parser{
			OpeningTag: openingTag,
			ClosingTag: closingTag,
		}

		if strings.HasSuffix(strings.TrimSpace(prompt), openingTag) {
			thinkingState.AddContent(openingTag)
		}
	}

	var toolParser *tools.Parser
	if len(req.Tools) > 0 && (builtinParser == nil || !builtinParser.HasToolSupport()) {
		toolParser = tools.NewParser(m.Template.Template, req.Tools)
	}

	type structuredOutputsState int
	const (
		structuredOutputsState_None structuredOutputsState = iota
		structuredOutputsState_ReadyToApply
		structuredOutputsState_Applying
	)

	ch := make(chan any)
	go func() {
		defer close(ch)

		structuredOutputsState := structuredOutputsState_None

		for {
			var tb strings.Builder

			currentFormat := req.Format
			// structured outputs via double request is enabled when:
			// 1. the model supports the thinking capability and
			// 2. it uses a built-in parser or our generic thinking parser

			// Note that the current approach does not work for (potential future)
			// non-thinking models that emit anything before actual content. This
			// current approach uses the transition from parsed thinking content to
			// parsed non-thinking content as the signal to turn constraining on

			if req.Format != nil && structuredOutputsState == structuredOutputsState_None && ((builtinParser != nil || thinkingState != nil) && slices.Contains(m.Capabilities(), model.CapabilityThinking)) {
				currentFormat = nil
			}

			// sets up new context given parent context per request
			ctx, cancel := context.WithCancel(c.Request.Context())
			err := r.Completion(ctx, llm.CompletionRequest{
				Prompt:      prompt,
				Images:      images,
				Format:      currentFormat,
				Options:     opts,
				Shift:       req.Shift == nil || *req.Shift,
				Truncate:    truncate,
				Logprobs:    req.Logprobs,
				TopLogprobs: req.TopLogprobs,
			}, func(r llm.CompletionResponse) {
				res := api.ChatResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC(),
					Message:   api.Message{Role: "assistant", Content: r.Content},
					Done:      r.Done,
					Metrics: api.Metrics{
						PromptEvalCount:    r.PromptEvalCount,
						PromptEvalDuration: r.PromptEvalDuration,
						EvalCount:          r.EvalCount,
						EvalDuration:       r.EvalDuration,
					},
					Logprobs: toAPILogprobs(r.Logprobs),
				}

				if r.Done {
					res.DoneReason = r.DoneReason.String()
					res.TotalDuration = time.Since(checkpointStart)
					res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
				}

				if builtinParser != nil {
					slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser input", "parser", m.Config.Parser, "content", r.Content)

					content, thinking, toolCalls, err := builtinParser.Add(r.Content, r.Done)
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}

					res.Message.Content = content
					res.Message.Thinking = thinking
					for i := range toolCalls {
						toolCalls[i].ID = toolCallId()
					}
					res.Message.ToolCalls = toolCalls

					tb.WriteString(thinking)
					// we are now receiving content from the model - we should start applying structured outputs
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && res.Message.Content != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						cancel()
						return
					}

					if res.Message.Content != "" || res.Message.Thinking != "" || len(res.Message.ToolCalls) > 0 || r.Done || len(res.Logprobs) > 0 {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser output", "parser", m.Config.Parser, "content", content, "thinking", thinking, "toolCalls", toolCalls, "done", r.Done)
						ch <- res
					} else {
						slog.Log(context.TODO(), logutil.LevelTrace, "builtin parser empty output", "parser", m.Config.Parser)
					}
					return
				}

				if thinkingState != nil {
					thinkingContent, remainingContent := thinkingState.AddContent(res.Message.Content)
					if thinkingContent == "" && remainingContent == "" && !r.Done {
						// need to accumulate more to decide what to send
						return
					}
					res.Message.Thinking = thinkingContent
					tb.WriteString(thinkingContent)
					// emit the collected thinking text before restarting with structured outputs and clear unstructured content
					// to avoid leaking mixed tokens like "</think>Hello"
					if structuredOutputsState == structuredOutputsState_None && req.Format != nil && tb.String() != "" && remainingContent != "" {
						structuredOutputsState = structuredOutputsState_ReadyToApply
						res.Message.Content = ""
						ch <- res
						cancel()
						return
					}
					res.Message.Content = remainingContent
				}

				if len(req.Tools) > 0 {
					toolCalls, content := toolParser.Add(res.Message.Content)
					if len(content) > 0 {
						res.Message.Content = content
					} else if len(toolCalls) > 0 {
						for i := range toolCalls {
							toolCalls[i].ID = toolCallId()
						}
						res.Message.ToolCalls = toolCalls
						res.Message.Content = ""
					} else if res.Message.Thinking != "" {
						// don't return, fall through to send
					} else {
						//  Send logprobs while content is being buffered by the parser for tool calls
						if len(res.Logprobs) > 0 && !r.Done {
							logprobRes := res
							logprobRes.Message.Content = ""
							logprobRes.Message.ToolCalls = nil
							ch <- logprobRes
						}

						if r.Done {
							res.Message.Content = toolParser.Content()
							ch <- res
						}
						return
					}
				}

				ch <- res
			})
			if err != nil {
				if structuredOutputsState == structuredOutputsState_ReadyToApply && strings.Contains(err.Error(), "context canceled") && c.Request.Context().Err() == nil {
					// only ignores error if it's a context cancellation due to setting structured outputs
				} else {
					var serr api.StatusError
					if errors.As(err, &serr) {
						ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
					} else {
						ch <- gin.H{"error": err.Error()}
					}
					return
				}
			}

			// ignored structured outputs cancellation falls through to here, start a new request with the structured outputs and updated prompt. use the
			if structuredOutputsState == structuredOutputsState_ReadyToApply {
				structuredOutputsState = structuredOutputsState_Applying
				msg := api.Message{
					Role:     "assistant",
					Thinking: tb.String(),
				}

				msgs = append(msgs, msg)
				prompt, _, err = chatPrompt(c.Request.Context(), m, r.Tokenize, opts, msgs, processedTools, req.Think, truncate)
				if err != nil {
					slog.Error("chat prompt error applying structured outputs", "error", err)
					ch <- gin.H{"error": err.Error()}
					return
				}
				// force constraining by terminating thinking header, the parser is already at this state
				// when the last message is thinking, the rendered for gpt-oss cannot disambiguate between having the
				// model continue thinking or ending thinking and outputting the final message.
				// TODO(parthsareen): consider adding prefill disambiguation logic to the renderer for structured outputs.
				if shouldUseHarmony(m) || (builtinParser != nil && m.Config.Parser == "harmony") {
					prompt += "<|end|><|start|>assistant<|channel|>final<|message|>"
				}
				continue
			}

			break
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var resp api.ChatResponse
		var toolCalls []api.ToolCall
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.ChatResponse:
				sbThinking.WriteString(t.Message.Thinking)
				sbContent.WriteString(t.Message.Content)
				resp = t
				if len(req.Tools) > 0 {
					toolCalls = append(toolCalls, t.Message.ToolCalls...)
				}
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		resp.Message.Content = sbContent.String()
		resp.Message.Thinking = sbThinking.String()
		resp.Logprobs = allLogprobs

		if len(toolCalls) > 0 {
			resp.Message.ToolCalls = toolCalls
		}

		c.JSON(http.StatusOK, resp)
		return
	}

	streamResponse(c, ch)
}

func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.Canceled):
		c.JSON(499, gin.H{"error": "request canceled"})
	case errors.Is(err, ErrMaxQueue):
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	if m.Config.ModelFamily == "qwen3" || model.ParseName(m.Name).Model == "deepseek-r1" {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}

		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				// TODO(drifkin): this is from before we added proper thinking support.
				// However, even if thinking is not enabled (and therefore we shouldn't
				// change the user output), we should probably perform this filtering
				// for all thinking models (not just qwen3 & deepseek-r1) since it tends
				// to save tokens and improve quality.
				thinkingState := &thinking.Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}

// handleImageGenerate handles image generation requests within GenerateHandler.
// This is called when the model has the Image capability.
func (s *Server) handleImageGenerate(c *gin.Context, req api.GenerateRequest, modelName string, checkpointStart time.Time) {
	// Validate image dimensions
	const maxDimension int32 = 4096
	if req.Width > maxDimension || req.Height > maxDimension {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("width and height must be <= %d", maxDimension)})
		return
	}

	// Schedule the runner for image generation
	runner, _, _, err := s.scheduleRunner(c.Request.Context(), modelName, []model.Capability{model.CapabilityImage}, nil, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	// Handle load-only request (empty prompt)
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	// Set headers for streaming response
	c.Header("Content-Type", "application/x-ndjson")

	// Get seed from options if provided
	var seed int64
	if s, ok := req.Options["seed"]; ok {
		switch v := s.(type) {
		case int:
			seed = int64(v)
		case int64:
			seed = v
		case float64:
			seed = int64(v)
		}
	}

	var streamStarted bool
	if err := runner.Completion(c.Request.Context(), llm.CompletionRequest{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  req.Steps,
		Seed:   seed,
	}, func(cr llm.CompletionResponse) {
		streamStarted = true
		res := api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			Done:      cr.Done,
		}

		if cr.TotalSteps > 0 {
			res.Completed = int64(cr.Step)
			res.Total = int64(cr.TotalSteps)
		}

		if cr.Image != "" {
			res.Image = cr.Image
		}

		if cr.Done {
			res.DoneReason = cr.DoneReason.String()
			res.Metrics.TotalDuration = time.Since(checkpointStart)
			res.Metrics.LoadDuration = checkpointLoaded.Sub(checkpointStart)
		}

		data, _ := json.Marshal(res)
		c.Writer.Write(append(data, '\n'))
		c.Writer.Flush()
	}); err != nil {
		// Only send JSON error if streaming hasn't started yet
		// (once streaming starts, headers are committed and we can't change status code)
		if !streamStarted {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
	}
}

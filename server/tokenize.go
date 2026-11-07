package server

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"maps"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner"
)

const (
	// maxTokenizeBodyBytes (32 MiB) bounds the text one request can make the
	// server tokenize.
	maxTokenizeBodyBytes = 32 << 20

	cloudErrTokenizeUnavailable = "tokenization is unavailable"
)

func (s *Server) TokenizeHandler(c *gin.Context) {
	var req api.TokenizeRequest
	if !bindTokenizeRequest(c, &req) {
		return
	}
	name := req.Model
	target := s.tokenizerTarget(c, name, req.Runner)
	if target == nil {
		return
	}
	switch {
	case target.cloudModel != "":
		req.Model, req.Runner = target.cloudModel, ""
		proxyCloudTokenize(c, req)
	case target.remote != nil:
		req.Model, req.Runner = target.remote.Config.RemoteModel, ""
		remoteTokenize(c, target.remote, func(client *api.Client) (any, error) {
			resp, err := client.Tokenize(c.Request.Context(), &req)
			if resp != nil {
				resp.Model = name
			}
			return resp, err
		})
	default:
		tokens, err := target.local.TokenizePrompt(c.Request.Context(), req.Prompt, req.AddSpecialTokens)
		if err != nil {
			writeTokenizeError(c, err)
			return
		}
		if tokens == nil {
			tokens = []int{}
		}
		c.JSON(http.StatusOK, api.TokenizeResponse{Model: name, Tokens: tokens, Count: len(tokens)})
	}
}

func (s *Server) DetokenizeHandler(c *gin.Context) {
	var req api.DetokenizeRequest
	if !bindTokenizeRequest(c, &req) {
		return
	}
	if req.Tokens == nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "tokens are required"})
		return
	}
	name := req.Model
	target := s.tokenizerTarget(c, name, req.Runner)
	if target == nil {
		return
	}
	switch {
	case target.cloudModel != "":
		req.Model, req.Runner = target.cloudModel, ""
		proxyCloudTokenize(c, req)
	case target.remote != nil:
		req.Model, req.Runner = target.remote.Config.RemoteModel, ""
		remoteTokenize(c, target.remote, func(client *api.Client) (any, error) {
			resp, err := client.Detokenize(c.Request.Context(), &req)
			if resp != nil {
				resp.Model = name
			}
			return resp, err
		})
	default:
		prompt, err := target.local.Detokenize(c.Request.Context(), req.Tokens)
		if err != nil {
			writeTokenizeError(c, err)
			return
		}
		c.JSON(http.StatusOK, api.DetokenizeResponse{Model: name, Prompt: prompt})
	}
}

func bindTokenizeRequest(c *gin.Context, req any) bool {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxTokenizeBodyBytes)
	if err := c.ShouldBindJSON(req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return false
	} else if err != nil {
		status := http.StatusBadRequest
		var sizeErr *http.MaxBytesError
		if errors.As(err, &sizeErr) {
			status = http.StatusRequestEntityTooLarge
		}
		c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
		return false
	}
	return true
}

// tokenizerTarget says where a model's tokenizer runs. Exactly one field is
// set.
type tokenizerTarget struct {
	cloudModel string
	remote     *Model
	local      llm.Tokenizer
}

// tokenizerTarget resolves name. It writes an error response and returns nil
// when the model cannot be tokenized. Tokenizing never loads a model.
func (s *Server) tokenizerTarget(c *gin.Context, name, runner string) *tokenizerTarget {
	if name == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return nil
	}
	modelRef, err := parseAndValidateModelRef(name)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", name))
		return nil
	}
	if modelRef.Source == modelSourceCloud {
		return &tokenizerTarget{cloudModel: modelRef.Base}
	}
	runner, err = normalizeRunner(runner)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return nil
	}
	m, err := s.getExistingModelForRunner(modelRef.Name.String(), runner)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", name)})
		} else {
			handleScheduleError(c, name, err)
		}
		return nil
	}
	if m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		if modelRef.Source == modelSourceLocal {
			c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", name)})
			return nil
		}
		return &tokenizerTarget{remote: m}
	}

	switch {
	case m.IsMLX():
		tok, err := s.mlxTokenizers.Tokenizer(m.Name, m.Runner, m.ManifestDigest)
		if errors.Is(err, mlxrunner.ErrNoTokenizer) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model '%s' does not support tokenization", name)})
			return nil
		}
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return nil
		}
		return &tokenizerTarget{local: tok}
	case m.isGGUF():
		// TODO: tokenize unloaded GGUF models with a CPU-only llama-server
		// (-ngl 0 --no-warmup, tiny context) started outside the scheduler and
		// stopped when idle, if there is demand.
		if s.sched != nil {
			if tok, ok := s.sched.loadedRunner(c.Request.Context(), m).(llm.Tokenizer); ok {
				return &tokenizerTarget{local: tok}
			}
		}
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model '%s' must be loaded to tokenize; load it with a generate or chat request first", name)})
		return nil
	default:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model '%s' does not support tokenization", name)})
		return nil
	}
}

func writeTokenizeError(c *gin.Context, err error) {
	status := http.StatusInternalServerError
	var statusErr api.StatusError
	if errors.As(err, &statusErr) {
		status = statusErr.StatusCode
	}
	c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
}

// tokenizeRouteMissing is the response for a peer that has no tokenize routes yet.
var tokenizeRouteMissing = gin.H{"error": "tokenization is not yet available for this model's host"}

// proxyCloudTokenize forwards to ollama.com, which answers with the same API
// once it serves these routes.
func proxyCloudTokenize(c *gin.Context, payload any) {
	header := c.Writer.Header().Clone()
	w := &missingRouteWriter{ResponseWriter: c.Writer}
	c.Writer = w
	proxyCloudJSONRequest(c, payload, cloudErrTokenizeUnavailable)
	c.Writer = w.ResponseWriter
	if w.missing {
		// Drop the peer's headers but keep ours, such as CORS.
		clear(c.Writer.Header())
		maps.Copy(c.Writer.Header(), header)
		c.JSON(http.StatusNotImplemented, tokenizeRouteMissing)
	}
}

// missingRouteWriter drops a 404 that is not a JSON API error: the peer has no
// such route.
type missingRouteWriter struct {
	gin.ResponseWriter
	missing bool
}

func (w *missingRouteWriter) WriteHeader(code int) {
	if code == http.StatusNotFound && !strings.HasPrefix(w.Header().Get("Content-Type"), "application/json") {
		w.missing = true
		return
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *missingRouteWriter) Write(b []byte) (int, error) {
	if w.missing {
		return len(b), nil
	}
	return w.ResponseWriter.Write(b)
}

func (w *missingRouteWriter) WriteString(s string) (int, error) {
	if w.missing {
		return len(s), nil
	}
	return w.ResponseWriter.WriteString(s)
}

func (w *missingRouteWriter) Flush() {
	if !w.missing {
		w.ResponseWriter.Flush()
	}
}

// remoteTokenize forwards to a remote model's Ollama host.
func remoteTokenize(c *gin.Context, m *Model, call func(*api.Client) (any, error)) {
	remoteURL := remoteModelURL(c, m, cloudErrTokenizeUnavailable)
	if remoteURL == nil {
		return
	}

	resp, err := call(api.NewClient(remoteURL, http.DefaultClient))
	var authErr api.AuthorizationError
	var statusErr api.StatusError
	switch {
	case errors.As(err, &authErr):
		sURL, err := signinURL()
		if err != nil {
			slog.Error(err.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "error getting authorization details"})
			return
		}
		c.JSON(authErr.StatusCode, gin.H{"error": "unauthorized", "signin_url": sURL})
	case errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusNotFound && isMissingRouteBody(statusErr.ErrorMessage):
		c.JSON(http.StatusNotImplemented, tokenizeRouteMissing)
	case errors.As(err, &statusErr):
		c.JSON(statusErr.StatusCode, gin.H{"error": statusErr.ErrorMessage})
	case err != nil:
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
	default:
		c.JSON(http.StatusOK, resp)
	}
}

// isMissingRouteBody reports a 404 body that [api.Client] could not decode as
// an API error: Ollama's router text or an HTML page.
func isMissingRouteBody(message string) bool {
	message = strings.TrimSpace(message)
	return message == "404 page not found" || strings.HasPrefix(message, "<")
}

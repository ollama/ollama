//go:build windows || darwin

// Package ui serves the Ollama desktop app.
package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/history"
	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/types/not"
	"github.com/ollama/ollama/app/ui/responses"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	ollamaAuth "github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/envconfig"
	_ "github.com/tkrajina/typescriptify-golang-structs/typescriptify"
)

//go:generate tscriptify -package=github.com/ollama/ollama/app/ui/responses -target=./app/codegen/gotypes.gen.ts responses/types.go
//go:generate npm --prefix ./app run build

var CORS = envconfig.Bool("OLLAMA_CORS")

// OllamaDotCom returns the URL for ollama.com, allowing override via environment variable
var OllamaDotCom = func() string {
	if url := os.Getenv("OLLAMA_DOT_COM_URL"); url != "" {
		return url
	}
	return "https://ollama.com"
}()

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) Written() bool {
	return r.code != 0
}

func (r *statusRecorder) WriteHeader(code int) {
	r.code = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Status() int {
	if r.code == 0 {
		return http.StatusOK
	}
	return r.code
}

func (r *statusRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

type Server struct {
	Logger  *slog.Logger
	Restart func()
	Token   string
	Store   *store.Store

	// Dev is true if the server is running in development mode
	Dev bool

	// Updater for checking and downloading updates
	Updater              *updater.Updater
	UpdateAvailableFunc  func()
	IntegrationInstalled func(string) bool
	ListCloudModels      func(context.Context) (*api.ListResponse, error)
	ExportChat           func(store.Chat) (*history.Result, error)
	ExportAllChats       func(context.Context, func(history.Progress) error) (*history.Result, error)
}

func (s *Server) log() *slog.Logger {
	if s.Logger == nil {
		return slog.Default()
	}
	return s.Logger
}

// ollamaProxy creates a reverse proxy handler to the Ollama server
func (s *Server) ollamaProxy() http.Handler {
	var (
		proxy   http.Handler
		proxyMu sync.Mutex
	)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyMu.Lock()
		p := proxy
		proxyMu.Unlock()

		if p == nil {
			proxyMu.Lock()
			if proxy == nil {
				var err error
				for i := range 2 {
					if i > 0 {
						s.log().Warn("ollama server not ready, retrying", "attempt", i+1)
						time.Sleep(1 * time.Second)
					}

					err = WaitForServer(context.Background(), 10*time.Second)
					if err == nil {
						break
					}
				}

				if err != nil {
					proxyMu.Unlock()
					s.log().Error("ollama server not ready after retries", "error", err)
					http.Error(w, "Ollama server is not ready", http.StatusServiceUnavailable)
					return
				}

				target := envconfig.ConnectableHost()
				s.log().Info("configuring ollama proxy", "target", target.String())

				newProxy := httputil.NewSingleHostReverseProxy(target)

				originalDirector := newProxy.Director
				newProxy.Director = func(req *http.Request) {
					originalDirector(req)
					req.Host = target.Host
					s.log().Debug("proxying request", "method", req.Method, "path", req.URL.Path, "target", target.Host)
				}

				newProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
					s.log().Error("proxy error", "error", err, "path", r.URL.Path, "target", target.String())
					http.Error(w, "proxy error: "+err.Error(), http.StatusBadGateway)
				}

				proxy = newProxy
				p = newProxy
			} else {
				p = proxy
			}
			proxyMu.Unlock()
		}

		p.ServeHTTP(w, r)
	})
}

type errHandlerFunc func(http.ResponseWriter, *http.Request) error

func (s *Server) Handler() http.Handler {
	handle := func(f errHandlerFunc) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Add CORS headers for dev work
			if CORS() {
				w.Header().Set("Access-Control-Allow-Origin", "*")
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, User-Agent, Accept, X-Requested-With")
				w.Header().Set("Access-Control-Allow-Credentials", "true")

				// Handle preflight requests
				if r.Method == "OPTIONS" {
					w.WriteHeader(http.StatusOK)
					return
				}
			}

			// Don't check for token in development mode
			if !s.Dev {
				cookie, err := r.Cookie("token")
				if err != nil {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}

				if cookie.Value != s.Token {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}
			}

			sw := &statusRecorder{ResponseWriter: w}

			log := s.log()
			level := slog.LevelInfo
			start := time.Now()
			requestID := fmt.Sprintf("%d", time.Now().UnixNano())

			defer func() {
				p := recover()
				if p != nil {
					log = log.With("panic", p, "request_id", requestID)
					level = slog.LevelError

					// Handle panic with user-friendly error
					if !sw.Written() {
						s.handleError(sw, fmt.Errorf("internal server error"))
					}
				}

				log.Log(r.Context(), level, "site.serveHTTP",
					"http.method", r.Method,
					"http.path", r.URL.Path,
					"http.pattern", r.Pattern,
					"http.status", sw.Status(),
					"http.d", time.Since(start),
					"request_id", requestID,
					"version", version.Version,
				)

				// let net/http.Server deal with panics
				if p != nil {
					panic(p)
				}
			}()

			w.Header().Set("X-Frame-Options", "DENY")
			w.Header().Set("X-Version", version.Version)
			w.Header().Set("X-Request-ID", requestID)

			ctx := r.Context()
			if err := f(sw, r); err != nil {
				if ctx.Err() != nil {
					return
				}
				level = slog.LevelError
				log = log.With("error", err)
				s.handleError(sw, err)
			}
		})
	}

	mux := http.NewServeMux()

	// CORS is handled in `handle`, but we have to match on OPTIONS to handle preflight requests
	mux.Handle("OPTIONS /", handle(func(w http.ResponseWriter, r *http.Request) error {
		return nil
	}))

	// API routes - handle first to take precedence
	mux.Handle("POST /api/v1/chat/{id}/export", handle(s.exportChat))
	mux.Handle("POST /api/v1/chats/export", handle(s.exportAllChats))
	mux.Handle("GET /api/v1/chats", handle(s.listChats))
	mux.Handle("GET /api/v1/chat/{id}", handle(s.getChat))
	mux.Handle("DELETE /api/v1/chat/{id}", handle(s.deleteChat))

	mux.Handle("GET /api/v1/inference-compute", handle(s.getInferenceCompute))
	mux.Handle("GET /api/v1/settings", handle(s.getSettings))
	mux.Handle("POST /api/v1/settings", handle(s.settings))
	mux.Handle("GET /api/v1/cloud", handle(s.getCloudSetting))
	mux.Handle("POST /api/v1/cloud", handle(s.cloudSetting))
	mux.Handle("GET /api/v1/models/cloud", handle(s.getCloudModels))
	mux.Handle("GET /api/v1/integrations", handle(s.getIntegrationStatuses))

	// Ollama proxy endpoints
	ollamaProxy := s.ollamaProxy()
	mux.Handle("GET /api/tags", ollamaProxy)
	mux.Handle("POST /api/show", ollamaProxy)
	mux.Handle("GET /api/version", ollamaProxy)
	mux.Handle("GET /api/status", ollamaProxy)
	mux.Handle("HEAD /api/version", ollamaProxy)
	mux.Handle("POST /api/me", ollamaProxy)
	mux.Handle("POST /api/signout", ollamaProxy)
	mux.Handle("GET /api/experimental/model-recommendations", ollamaProxy)

	// Only navigations fall back to the React app; retired writes return 405.
	mux.Handle("GET /", s.appHandler())

	return mux
}

func (s *Server) getIntegrationStatuses(w http.ResponseWriter, _ *http.Request) error {
	isInstalled := s.IntegrationInstalled
	if isInstalled == nil {
		isInstalled = launch.IsIntegrationInstalled
	}

	type integrationStatus struct {
		ID          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description"`
		Installed   *bool  `json:"installed,omitempty"`
		Action      string `json:"action"`
		Command     string `json:"command,omitempty"`
	}

	infos := launch.ListIntegrationInfos()
	statuses := make([]integrationStatus, 0, len(infos)+2)
	claudeDesktopInstalled := isInstalled("claude-desktop")
	statuses = append(statuses, integrationStatus{
		ID:          "claude-desktop",
		Name:        "Claude Code (Desktop)",
		Description: "Use Ollama models in Claude Desktop",
		Installed:   &claudeDesktopInstalled,
		Action:      "connect",
	})

	byName := make(map[string]launch.IntegrationInfo, len(infos))
	for _, info := range infos {
		byName[info.Name] = info
	}
	seen := map[string]bool{"chatgpt": true}
	launcherMenuOrder := []string{"claude", "codex", "openclaw", "opencode", "hermes", "hermes-desktop", "droid", "pi", "cline"}
	orderedInfos := make([]launch.IntegrationInfo, 0, len(infos))
	for _, name := range launcherMenuOrder {
		if info, ok := byName[name]; ok {
			orderedInfos = append(orderedInfos, info)
			seen[name] = true
		}
	}
	for _, info := range infos {
		if !seen[info.Name] {
			orderedInfos = append(orderedInfos, info)
		}
	}

	for _, info := range orderedInfos {
		installed := isInstalled(info.Name)
		statuses = append(statuses, integrationStatus{
			ID:          info.Name,
			Name:        info.DisplayName,
			Description: info.Description,
			Installed:   &installed,
			Action:      "copy",
			Command:     "ollama launch " + info.Name,
		})
	}

	statuses = append(statuses, integrationStatus{
		ID:          "terminal",
		Name:        "Terminal",
		Description: "Run local models from your terminal",
		Action:      "copy",
		Command:     "ollama",
	})

	return json.NewEncoder(w).Encode(statuses)
}

// handleError renders appropriate error responses based on request type
func (s *Server) handleError(w http.ResponseWriter, e error) {
	// Preserve CORS headers for API requests
	if CORS() {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, User-Agent, Accept, X-Requested-With")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(map[string]string{"error": e.Error()})
}

// userAgentTransport is a custom RoundTripper that adds the User-Agent header to all requests
type userAgentTransport struct {
	base http.RoundTripper
}

func (t *userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request to avoid mutating the original
	r := req.Clone(req.Context())
	r.Header.Set("User-Agent", userAgent())
	return t.base.RoundTrip(r)
}

// httpClient returns an HTTP client that automatically adds the User-Agent header
func (s *Server) httpClient() *http.Client {
	return userAgentHTTPClient(10 * time.Second)
}

func userAgentHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &userAgentTransport{
			base: http.DefaultTransport,
		},
	}
}

// doSelfSigned sends a self-signed request to the ollama.com API
func (s *Server) doSelfSigned(ctx context.Context, method, path string) (*http.Response, error) {
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	// Form the string to sign: METHOD,PATH?ts=TIMESTAMP
	signString := fmt.Sprintf("%s,%s?ts=%s", method, path, timestamp)
	signature, err := ollamaAuth.Sign(ctx, []byte(signString))
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	endpoint := fmt.Sprintf("%s%s?ts=%s", OllamaDotCom, path, timestamp)
	req, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))

	return s.httpClient().Do(req)
}

// UserData fetches user data from ollama.com API for the current ollama key
func (s *Server) UserData(ctx context.Context) (*api.UserResponse, error) {
	resp, err := s.doSelfSigned(ctx, http.MethodPost, "/api/me")
	if err != nil {
		return nil, fmt.Errorf("failed to call ollama.com/api/me: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var user api.UserResponse
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, fmt.Errorf("failed to parse user response: %w", err)
	}

	user.AvatarURL = fmt.Sprintf("%s/%s", OllamaDotCom, user.AvatarURL)

	storeUser := store.User{
		Name:  user.Name,
		Email: user.Email,
		Plan:  user.Plan,
	}
	if err := s.Store.SetUser(storeUser); err != nil {
		s.log().Warn("failed to cache user data", "error", err)
	}

	return &user, nil
}

// WaitForServer waits for the Ollama server to be ready
func WaitForServer(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c, err := api.ClientFromEnvironment()
		if err != nil {
			return err
		}
		if _, err := c.Version(ctx); err == nil {
			slog.Debug("ollama server is ready")
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return errors.New("timeout waiting for Ollama server to be ready")
}

func (s *Server) listChats(w http.ResponseWriter, r *http.Request) error {
	chats, err := s.Store.Chats()
	if err != nil {
		return err
	}

	chatInfos := make([]responses.ChatInfo, len(chats))
	for i, chat := range chats {
		chatInfos[i] = chatInfoFromChat(chat)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responses.ChatsResponse{ChatInfos: chatInfos})
	return nil
}

func (s *Server) getChat(w http.ResponseWriter, r *http.Request) error {
	cid := r.PathValue("id")

	if cid == "" {
		return fmt.Errorf("chat ID is required")
	}

	chat, err := s.Store.Chat(cid)
	if err != nil {
		return err
	}

	// fill missing tool_name on tool messages (from previous tool_calls) so labels don’t flip after reload.
	if chat != nil && len(chat.Messages) > 0 {
		for i := range chat.Messages {
			if chat.Messages[i].Role == "tool" && chat.Messages[i].ToolName == "" && chat.Messages[i].ToolResult != nil {
				for j := i - 1; j >= 0; j-- {
					if chat.Messages[j].Role == "assistant" && len(chat.Messages[j].ToolCalls) > 0 {
						last := chat.Messages[j].ToolCalls[len(chat.Messages[j].ToolCalls)-1]
						if last.Function.Name != "" {
							chat.Messages[i].ToolName = last.Function.Name
						}
						break
					}
				}
			}
		}
	}

	// Older chats saved the browser state in tool results instead of on the chat.
	if len(chat.BrowserState) == 0 {
		for i := len(chat.Messages) - 1; i >= 0; i-- {
			result := chat.Messages[i].ToolResult
			if result == nil {
				continue
			}
			var state responses.BrowserStateData
			if err := json.Unmarshal(*result, &state); err == nil && (len(state.PageStack) > 0 || len(state.URLToPage) > 0) {
				chat.BrowserState = *result
				break
			}
		}
	}

	data := responses.ChatResponse{
		Chat: *chat,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
	return nil
}

func (s *Server) exportChat(w http.ResponseWriter, r *http.Request) error {
	if s.ExportChat == nil {
		return errors.New("Export is unavailable in this window")
	}
	chat, err := s.Store.Chat(r.PathValue("id"))
	if err != nil {
		return err
	}
	result, err := s.ExportChat(*chat)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(result)
}

func (s *Server) exportAllChats(w http.ResponseWriter, r *http.Request) error {
	if s.ExportAllChats == nil {
		return errors.New("Export is unavailable in this window")
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	encoder := json.NewEncoder(w)
	send := func(value any) error {
		if err := encoder.Encode(value); err != nil {
			return err
		}
		return http.NewResponseController(w).Flush()
	}
	result, err := s.ExportAllChats(r.Context(), func(progress history.Progress) error {
		return send(progress)
	})
	if err := r.Context().Err(); err != nil {
		return err
	}
	if err != nil {
		return send(map[string]string{"error": err.Error()})
	}
	return send(result)
}

func (s *Server) deleteChat(w http.ResponseWriter, r *http.Request) error {
	cid := r.PathValue("id")
	if cid == "" {
		return fmt.Errorf("chat ID is required")
	}

	if err := s.Store.DeleteChat(cid); err != nil {
		if errors.Is(err, not.Found) {
			w.WriteHeader(http.StatusNotFound)
		}
		return fmt.Errorf("failed to delete chat: %w", err)
	}

	w.WriteHeader(http.StatusOK)
	return nil
}

func chatInfoFromChat(chat store.Chat) responses.ChatInfo {
	userExcerpt := ""
	var updatedAt time.Time

	for _, msg := range chat.Messages {
		// extract the first user message as the user excerpt
		if msg.Role == "user" && userExcerpt == "" {
			userExcerpt = msg.Content
		}
		// update the updated at time
		if msg.UpdatedAt.After(updatedAt) {
			updatedAt = msg.UpdatedAt
		}
	}

	return responses.ChatInfo{
		ID:          chat.ID,
		Title:       chat.Title,
		UserExcerpt: userExcerpt,
		CreatedAt:   chat.CreatedAt,
		UpdatedAt:   updatedAt,
	}
}

func (s *Server) getSettings(w http.ResponseWriter, r *http.Request) error {
	settings, err := s.Store.Settings()
	if err != nil {
		return fmt.Errorf("failed to load settings: %w", err)
	}

	// set default models directory if not set
	if settings.Models == "" {
		settings.Models = envconfig.Models()
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(responses.SettingsResponse{
		Settings: settings,
	})
}

func (s *Server) settings(w http.ResponseWriter, r *http.Request) error {
	old, err := s.Store.Settings()
	if err != nil {
		return fmt.Errorf("failed to load settings: %w", err)
	}

	var request struct {
		store.Settings
		OnboardingVersion *int
		ClaudeDesktopUsed *bool
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}

	settings := request.Settings
	if request.OnboardingVersion == nil {
		settings.OnboardingVersion = old.OnboardingVersion
	} else {
		settings.OnboardingVersion = *request.OnboardingVersion
	}
	if request.ClaudeDesktopUsed == nil {
		settings.ClaudeDesktopUsed = old.ClaudeDesktopUsed
	} else {
		settings.ClaudeDesktopUsed = *request.ClaudeDesktopUsed
	}

	if err := s.Store.SetSettings(settings); err != nil {
		return fmt.Errorf("failed to save settings: %w", err)
	}
	saved, err := s.Store.Settings()
	if err != nil {
		return fmt.Errorf("failed to load saved settings: %w", err)
	}
	settings.OnboardingVersion = saved.OnboardingVersion
	settings.CodexDesktopUsed = saved.CodexDesktopUsed

	// Handle auto-update toggle changes
	if old.AutoUpdateEnabled != settings.AutoUpdateEnabled {
		if !settings.AutoUpdateEnabled {
			// Auto-update disabled: cancel any ongoing download
			if s.Updater != nil {
				s.Updater.CancelOngoingDownload()
			}
		} else {
			// Auto-update re-enabled: show notification if update is already staged, or trigger immediate check
			if (updater.IsUpdatePending() || updater.UpdateDownloaded) && s.UpdateAvailableFunc != nil {
				s.UpdateAvailableFunc()
			} else if s.Updater != nil {
				// Trigger the background checker to run immediately
				s.Updater.TriggerImmediateCheck()
			}
		}
	}

	if old.ContextLength != settings.ContextLength ||
		old.Models != settings.Models ||
		old.Expose != settings.Expose {
		s.Restart()
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(responses.SettingsResponse{
		Settings: settings,
	})
}

func (s *Server) cloudSetting(w http.ResponseWriter, r *http.Request) error {
	var req struct {
		Enabled bool `json:"enabled"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}

	if err := s.Store.SetCloudEnabled(req.Enabled); err != nil {
		return fmt.Errorf("failed to persist cloud setting: %w", err)
	}

	s.Restart()

	return s.writeCloudStatus(w)
}

func (s *Server) getCloudSetting(w http.ResponseWriter, r *http.Request) error {
	return s.writeCloudStatus(w)
}

func (s *Server) writeCloudStatus(w http.ResponseWriter) error {
	disabled, source, err := s.Store.CloudStatus()
	if err != nil {
		return fmt.Errorf("failed to load cloud status: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(map[string]any{
		"disabled": disabled,
		"source":   source,
	})
}

func (s *Server) getCloudModels(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Content-Type", "application/json")
	disabled, _, err := s.Store.CloudStatus()
	if err != nil {
		return fmt.Errorf("failed to load cloud status: %w", err)
	}
	if disabled {
		return json.NewEncoder(w).Encode(api.ListResponse{Models: []api.ListModelResponse{}})
	}

	list := s.ListCloudModels
	if list == nil {
		list = s.listCloudModels
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	models, err := list(ctx)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && (authErr.StatusCode == http.StatusUnauthorized || authErr.StatusCode == http.StatusForbidden) {
			return json.NewEncoder(w).Encode(api.ListResponse{Models: []api.ListModelResponse{}})
		}
		return fmt.Errorf("failed to list cloud models: %w", err)
	}
	if models == nil {
		models = &api.ListResponse{Models: []api.ListModelResponse{}}
	}

	return json.NewEncoder(w).Encode(models)
}

func (s *Server) listCloudModels(ctx context.Context) (*api.ListResponse, error) {
	resp, err := s.doSelfSigned(ctx, http.MethodGet, "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, api.AuthorizationError{StatusCode: resp.StatusCode}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama.com/api/tags returned %s", resp.Status)
	}

	var models api.ListResponse
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return nil, fmt.Errorf("failed to parse cloud models: %w", err)
	}
	return &models, nil
}

func (s *Server) getInferenceCompute(w http.ResponseWriter, r *http.Request) error {
	ctx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
	defer cancel()
	info, err := server.GetInferenceInfo(ctx)
	if err != nil {
		s.log().Error("failed to get inference info", "error", err)
		return fmt.Errorf("failed to get inference info: %w", err)
	}

	inferenceComputes := make([]responses.InferenceCompute, len(info.Computes))
	for i, ic := range info.Computes {
		inferenceComputes[i] = responses.InferenceCompute{
			Library: ic.Library,
			Variant: ic.Variant,
			Compute: ic.Compute,
			Driver:  ic.Driver,
			Name:    ic.Name,
			VRAM:    ic.VRAM,
		}
	}

	response := responses.InferenceComputeResponse{
		InferenceComputes:    inferenceComputes,
		DefaultContextLength: info.DefaultContextLength,
	}

	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(response)
}

func userAgent() string {
	buildinfo, _ := debug.ReadBuildInfo()

	version := buildinfo.Main.Version
	if version == "(devel)" {
		// When using `go run .` the version is "(devel)". This is seen
		// as an invalid version by ollama.com and so it defaults to
		// "needs upgrade" for some requests, such as pulls. These
		// checks can be skipped by using the special version "v0.0.0",
		// so we set it to that here.
		version = "v0.0.0"
	}

	return fmt.Sprintf("ollama/%s (%s %s) app/%s Go/%s",
		version,
		runtime.GOARCH,
		runtime.GOOS,
		version,
		runtime.Version(),
	)
}

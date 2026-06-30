package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	coreagent "github.com/ollama/ollama/agent"
	agenttools "github.com/ollama/ollama/agent/tools"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	agentchat "github.com/ollama/ollama/cmd/tui/chat"
	"github.com/ollama/ollama/format"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
)

type agentTUIOptions struct {
	Model               string
	OpenModelPicker     bool
	System              string
	Format              string
	Options             map[string]any
	Think               *api.ThinkValue
	KeepAlive           *api.Duration
	ContextWindowTokens int
	AllowAllTools       bool
	MultiModal          bool
}

func registerAgentFlags(cmd *cobra.Command) {
	cmd.Flags().String("model", "", "Model to use")
	cmd.Flags().String("keepalive", "", "Duration to keep a model loaded (e.g. 5m)")
	cmd.Flags().String("format", "", "Response format (e.g. json)")
	cmd.Flags().String("think", "", "Enable thinking mode: true/false or high/medium/low for supported models")
	cmd.Flags().Lookup("think").NoOptDefVal = "true"
	cmd.Flags().Bool("auto-approve-tools", false, "Allow agent tools to run without prompting")
	cmd.Flags().Bool("yolo", false, "Alias for --auto-approve-tools")
}

func AgentHandler(cmd *cobra.Command, _ []string) error {
	opts := agentTUIOptions{
		Model:   strings.TrimSpace(config.LastModel()),
		Options: map[string]any{},
	}
	thinkExplicit, err := applyAgentFlags(cmd, &opts)
	if err != nil {
		return err
	}

	if strings.TrimSpace(opts.Model) == "" {
		opts.OpenModelPicker = true
	} else if cmd.Flags().Lookup("model") == nil || !cmd.Flags().Lookup("model").Changed {
		opts.OpenModelPicker = true
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if strings.TrimSpace(opts.Model) != "" && !opts.OpenModelPicker {
		info, err := prepareAgentModel(cmd, client, &opts, thinkExplicit)
		if err != nil {
			if handleCloudAuthorizationError(err) {
				return nil
			}
			return err
		}
		opts.System = info.System
	}

	if err := GenerateAgentTUI(cmd, client, opts); err != nil {
		if handleCloudAuthorizationError(err) {
			return nil
		}
		return fmt.Errorf("error running agent: %w", err)
	}
	return nil
}

func applyAgentFlags(cmd *cobra.Command, opts *agentTUIOptions) (bool, error) {
	if flag := cmd.Flags().Lookup("model"); flag != nil && flag.Changed {
		modelName, err := cmd.Flags().GetString("model")
		if err != nil {
			return false, err
		}
		modelName = strings.TrimSpace(modelName)
		if modelName == "" {
			return false, errors.New("--model cannot be empty")
		}
		opts.Model = modelName
		opts.OpenModelPicker = false
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return false, err
	}
	opts.Format = format

	thinkExplicit := false
	thinkFlag := cmd.Flags().Lookup("think")
	if thinkFlag != nil && thinkFlag.Changed {
		thinkExplicit = true
		thinkStr, err := cmd.Flags().GetString("think")
		if err != nil {
			return false, err
		}
		switch thinkStr {
		case "", "true":
			opts.Think = &api.ThinkValue{Value: true}
		case "false":
			opts.Think = &api.ThinkValue{Value: false}
		case "high", "medium", "low", "max":
			opts.Think = &api.ThinkValue{Value: thinkStr}
		default:
			return false, fmt.Errorf("invalid value for --think: %q (must be true, false, high, medium, low, or max)", thinkStr)
		}
	}

	keepAlive, err := cmd.Flags().GetString("keepalive")
	if err != nil {
		return false, err
	}
	if keepAlive != "" {
		d, err := time.ParseDuration(keepAlive)
		if err != nil {
			return false, err
		}
		opts.KeepAlive = &api.Duration{Duration: d}
	}

	autoApprove, err := cmd.Flags().GetBool("auto-approve-tools")
	if err != nil {
		return false, err
	}
	yolo, err := cmd.Flags().GetBool("yolo")
	if err != nil {
		return false, err
	}
	opts.AllowAllTools = autoApprove || yolo
	return thinkExplicit, nil
}

func prepareAgentModel(cmd *cobra.Command, client *api.Client, opts *agentTUIOptions, thinkExplicit bool) (*api.ShowResponse, error) {
	requestedCloud := modelref.HasExplicitCloudSource(opts.Model)
	info, err := func() (*api.ShowResponse, error) {
		info, err := client.Show(cmd.Context(), &api.ShowRequest{Model: opts.Model})
		var se api.StatusError
		if errors.As(err, &se) && se.StatusCode == http.StatusNotFound {
			if requestedCloud {
				return nil, err
			}
			if err := PullHandler(cmd, []string{opts.Model}); err != nil {
				return nil, err
			}
			return client.Show(cmd.Context(), &api.ShowRequest{Model: opts.Model})
		}
		return info, err
	}()
	if err != nil {
		return nil, err
	}

	ensureCloudStub(cmd.Context(), client, opts.Model)
	opts.Think, err = inferThinkingOption(&info.Capabilities, &runOptions{Model: opts.Model, Think: opts.Think}, thinkExplicit)
	if err != nil {
		return nil, err
	}
	opts.MultiModal = showResponseSupportsMultimodal(info)
	opts.ContextWindowTokens = showResponseContextWindow(info)
	return info, nil
}

func GenerateAgentTUI(cmd *cobra.Command, client *api.Client, opts agentTUIOptions) error {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}

	registry := agentToolsRegistry(cmd.Context(), client, opts.Model)
	systemPrompt := agentSystemPrompt(opts.Model, opts.System, "")

	_, err = agentchat.Run(cmd.Context(), agentchat.Options{
		Model:           opts.Model,
		OpenModelPicker: opts.OpenModelPicker,
		Client:          client,
		Tools:           registry,
		ToolRegistryForModel: func(ctx context.Context, model string) *coreagent.Registry {
			return agentToolsRegistry(ctx, client, model)
		},
		MultiModalForModel: func(ctx context.Context, model string) bool {
			return agentModelSupportsMultimodal(ctx, client, model)
		},
		ModelOptions: func(ctx context.Context) ([]agentchat.ModelOption, error) {
			return agentModelOptions(ctx, client)
		},
		OnModelSelected: func(_ context.Context, model string) error {
			return config.SetLastModel(model)
		},
		SystemPromptForModel: func(ctx context.Context, model string, registry *coreagent.Registry) string {
			return agentSystemPrompt(model, agentSystemFromShow(ctx, client, model), "")
		},
		SystemPrompt:        systemPrompt,
		WorkingDir:          cwd,
		Format:              opts.Format,
		Options:             opts.Options,
		Think:               opts.Think,
		KeepAlive:           opts.KeepAlive,
		MultiModal:          opts.MultiModal,
		AllowAllTools:       opts.AllowAllTools,
		ContextWindowTokens: opts.ContextWindowTokens,
		Compactor: &coreagent.SimpleCompactor{
			Client:  client,
			Options: coreagent.CompactionOptions{ContextWindowTokens: opts.ContextWindowTokens},
		},
		ContextWindowTokensForModel: func(ctx context.Context, model string, fallback int) int {
			return agentContextWindowForModel(ctx, client, model, fallback)
		},
		PreloadModel: func(ctx context.Context, model string, think *api.ThinkValue) error {
			return preloadAgentModelIfLocal(ctx, client, opts, model, think)
		},
		CheckCloudModel: func(ctx context.Context, model, requiredPlan string) error {
			return ensureCloudModelAccess(ctx, client, model, requiredPlan)
		},
		OpenBrowser: launch.OpenBrowser,
		PollCloudAuth: func(ctx context.Context) (string, bool) {
			user, err := client.Whoami(ctx)
			if err != nil || user == nil || user.Name == "" {
				return "", false
			}
			return user.Name, true
		},
	})
	return err
}

func agentSystemPrompt(modelName string, modelSystem string, extra string) string {
	return agentSystemPromptAt(time.Now(), modelName, modelSystem, extra)
}

func agentSystemPromptAt(now time.Time, modelName string, modelSystem string, extra string) string {
	var parts []string
	parts = append(parts, agentDefaultSystemPrompt(now, modelName))
	if strings.TrimSpace(modelSystem) != "" {
		parts = append(parts, strings.TrimSpace(modelSystem))
	}
	if strings.TrimSpace(extra) != "" {
		parts = append(parts, strings.TrimSpace(extra))
	}
	return strings.Join(parts, "\n\n")
}

func agentDefaultSystemPrompt(now time.Time, modelName string) string {
	date := now.Format("Monday, January 2, 2006")
	shellName := "bash"
	if runtime.GOOS == "windows" {
		shellName = "PowerShell"
	}
	return strings.Join([]string{
		"You are running in Ollama, in a harness to help the user accomplish tasks, and the model is " + modelName + ".",
		"",
		"Current date: " + date + ".",
		"",
		"Be concise, practical, and action-oriented. Use tools when they materially help. Verify current or fast-changing facts with web tools when available; otherwise state uncertainty.",
		"",
		"Use " + shellName + " carefully. Prefer read-only inspection first. Stay within the current working directory unless explicitly asked. Surface intent before risky actions such as writes, deletes, moves, installs, git state changes, service changes, sudo, secrets access, network scripts, or commands outside the working directory. Request approval when required and do not work around denied approvals.",
		"",
		"Tell the user about meaningful changes, verification, failures, blockers, assumptions, and risks. Summarize routine tool output instead of dumping it.",
	}, "\n")
}

func agentSystemFromShow(ctx context.Context, client *api.Client, modelName string) string {
	if client == nil || strings.TrimSpace(modelName) == "" {
		return ""
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not load model system prompt: %v\n", err)
		return ""
	}
	return resp.System
}

func agentToolsRegistry(ctx context.Context, client *api.Client, modelName string) *coreagent.Registry {
	supportsTools, err := agentModelSupportsTools(ctx, client, modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not check model capabilities: %v\n", err)
	}
	if !supportsTools {
		return nil
	}

	registry := &coreagent.Registry{}
	if os.Getenv("OLLAMA_AGENT_DISABLE_SHELL") == "" {
		registry.Register(&agenttools.Bash{})
	}
	registry.Register(&agenttools.Read{})
	registry.Register(&agenttools.Edit{})

	if os.Getenv("OLLAMA_AGENT_DISABLE_WEBSEARCH") == "" {
		if disabled, known := agentCloudStatusDisabled(ctx, client); !known || !disabled {
			registry.Register(&agenttools.WebSearch{})
			registry.Register(&agenttools.WebFetch{})
		} else {
			fmt.Fprintf(os.Stderr, "%s\n", internalcloud.DisabledError("web search is unavailable"))
		}
	}
	return registry
}

func agentModelSupportsTools(ctx context.Context, client *api.Client, modelName string) (bool, error) {
	if client == nil || strings.TrimSpace(modelName) == "" {
		return false, nil
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return false, err
	}
	return slices.Contains(resp.Capabilities, model.CapabilityTools), nil
}

func agentModelSupportsMultimodal(ctx context.Context, client *api.Client, modelName string) bool {
	if client == nil || strings.TrimSpace(modelName) == "" {
		return false
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not check model capabilities: %v\n", err)
		return false
	}
	return showResponseSupportsMultimodal(resp)
}

func showResponseSupportsMultimodal(resp *api.ShowResponse) bool {
	if resp == nil {
		return false
	}
	if slices.Contains(resp.Capabilities, model.CapabilityVision) || slices.Contains(resp.Capabilities, model.CapabilityAudio) {
		return true
	}
	if len(resp.ProjectorInfo) != 0 {
		return true
	}
	for key := range resp.ModelInfo {
		if strings.Contains(key, ".vision.") {
			return true
		}
	}
	return false
}

func agentContextWindowForModel(ctx context.Context, client *api.Client, modelName string, fallback int) int {
	if client == nil || strings.TrimSpace(modelName) == "" {
		return fallback
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return fallback
	}
	if tokens := showResponseContextWindow(resp); tokens > 0 {
		return tokens
	}
	return fallback
}

func showResponseContextWindow(resp *api.ShowResponse) int {
	if resp == nil {
		return 0
	}
	if resp.Details.ContextLength > 0 {
		return resp.Details.ContextLength
	}
	for _, key := range []string{
		"general.context_length",
		"llama.context_length",
		"qwen2.context_length",
	} {
		if n, ok := numericModelInfo(resp.ModelInfo[key]); ok {
			return n
		}
	}
	return 0
}

func numericModelInfo(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, v > 0
	case int64:
		return int(v), v > 0
	case float64:
		return int(v), v > 0
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		return n, err == nil && n > 0
	default:
		return 0, false
	}
}

func preloadAgentModelIfLocal(ctx context.Context, client *api.Client, opts agentTUIOptions, modelName string, think *api.ThinkValue) error {
	modelName = strings.TrimSpace(modelName)
	if client == nil || modelName == "" {
		return nil
	}
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return err
	}
	if info.RemoteHost != "" || modelref.HasExplicitCloudSource(modelName) {
		return nil
	}
	return client.Generate(ctx, &api.GenerateRequest{
		Model:     modelName,
		KeepAlive: opts.KeepAlive,
		Think:     think,
	}, func(api.GenerateResponse) error {
		return nil
	})
}

func agentModelOptions(ctx context.Context, client *api.Client) ([]agentchat.ModelOption, error) {
	if client == nil {
		return nil, errors.New("model picker requires an API client")
	}

	list, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	seen := make(map[string]struct{})
	var options []agentchat.ModelOption
	add := func(name, description string, recommended bool, requiredPlan string, cloud bool) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		key := strings.ToLower(name)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		options = append(options, agentchat.ModelOption{
			Name:         name,
			Description:  strings.TrimSpace(description),
			Recommended:  recommended,
			RequiredPlan: requiredPlan,
			Cloud:        cloud,
		})
	}

	if disabled, known := agentCloudStatusDisabled(ctx, client); !known || !disabled {
		if recs, err := client.ModelRecommendationsExperimental(ctx); err == nil {
			for _, rec := range recs.Recommendations {
				name := strings.TrimSpace(rec.Model)
				if !modelref.HasExplicitCloudSource(name) {
					continue
				}
				add(name, agentRecommendationDescription(rec), true, strings.TrimSpace(rec.RequiredPlan), true)
			}
		}
	}

	local := slices.Clone(list.Models)
	slices.SortStableFunc(local, func(a, b api.ListModelResponse) int {
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})
	for _, model := range local {
		name := strings.TrimSpace(model.Name)
		if name == "" {
			name = strings.TrimSpace(model.Model)
		}
		name = strings.TrimSuffix(name, ":latest")
		if modelref.HasExplicitCloudSource(name) {
			add(name, agentCloudModelDescription(model), false, "", true)
			continue
		}
		add(name, agentLocalModelDescription(model), false, "", false)
	}

	badges, signInURLs := cloudAvailabilityBadges(ctx, client, options)
	for i := range options {
		options[i].AvailabilityBadge = badges[options[i].Name]
		options[i].SignInURL = signInURLs[options[i].Name]
	}
	return options, nil
}

func cloudAvailabilityBadges(ctx context.Context, client *api.Client, options []agentchat.ModelOption) (map[string]string, map[string]string) {
	badges := make(map[string]string)
	signInURLs := make(map[string]string)
	hasCloud := false
	for _, opt := range options {
		if opt.Cloud {
			hasCloud = true
			break
		}
	}
	if !hasCloud {
		return badges, signInURLs
	}

	if disabled, known := agentCloudStatusDisabled(ctx, client); known && disabled {
		return badges, signInURLs
	}

	whoamiCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	user, err := client.Whoami(whoamiCtx)
	if err != nil {
		var authErr api.AuthorizationError
		signInURL := ""
		if errors.As(err, &authErr) && authErr.SigninURL != "" {
			signInURL = authErr.SigninURL
		}
		for _, opt := range options {
			if opt.Cloud {
				badges[opt.Name] = "Sign in required"
				if signInURL != "" {
					signInURLs[opt.Name] = signInURL
				}
			}
		}
		return badges, signInURLs
	}

	signedIn := user != nil && user.Name != ""
	for _, opt := range options {
		if !opt.Cloud {
			continue
		}
		if !signedIn {
			badges[opt.Name] = "Sign in required"
		} else if opt.RequiredPlan != "" && !launch.PlanSatisfies(user.Plan, opt.RequiredPlan) {
			badges[opt.Name] = "Upgrade required"
		}
	}
	return badges, signInURLs
}

func agentRecommendationDescription(rec api.ModelRecommendation) string {
	var parts []string
	if description := strings.TrimSpace(rec.Description); description != "" {
		parts = append(parts, description)
	} else {
		parts = append(parts, "cloud")
	}
	if rec.ContextLength > 0 {
		parts = append(parts, format.HumanNumber(uint64(rec.ContextLength))+" ctx")
	}
	return strings.Join(parts, " - ")
}

func agentLocalModelDescription(model api.ListModelResponse) string {
	desc := agentModelArchDescription(model)
	if desc == "" {
		return "local"
	}
	return "local - " + desc
}

func agentCloudModelDescription(model api.ListModelResponse) string {
	return agentModelArchDescription(model)
}

func agentModelArchDescription(model api.ListModelResponse) string {
	var details []string
	if model.Details.Family != "" {
		details = append(details, model.Details.Family)
	}
	if ps := humanizedParameterSize(model.Details.ParameterSize); ps != "" {
		details = append(details, ps)
	}
	if model.Details.QuantizationLevel != "" {
		details = append(details, model.Details.QuantizationLevel)
	}
	var parts []string
	if len(details) > 0 {
		parts = append(parts, strings.Join(details, " "))
	}
	if model.Details.ContextLength > 0 {
		parts = append(parts, format.HumanNumber(uint64(model.Details.ContextLength))+" ctx")
	}
	return strings.Join(parts, " - ")
}

func humanizedParameterSize(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return format.HumanNumber(uint64(f))
	}
	return s
}

func agentCloudStatusDisabled(ctx context.Context, client *api.Client) (disabled bool, known bool) {
	if internalcloud.Disabled() {
		return true, true
	}

	status, err := client.CloudStatusExperimental(ctx)
	if err != nil {
		var statusErr api.StatusError
		if errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusNotFound {
			return false, false
		}
		return false, false
	}
	return status.Cloud.Disabled, true
}

func ensureCloudModelAccess(ctx context.Context, client *api.Client, modelName, requiredPlan string) error {
	if client == nil {
		return errors.New("no API client available")
	}
	if disabled, known := agentCloudStatusDisabled(ctx, client); known && disabled {
		return errors.New("remote inference is unavailable")
	}
	user, err := client.Whoami(ctx)
	if err != nil {
		return err
	}
	if user != nil && user.Name != "" {
		if requiredPlan != "" && !launch.PlanSatisfies(user.Plan, requiredPlan) {
			return fmt.Errorf("plan upgrade required: %s needs plan %s, you have %s", modelName, requiredPlan, user.Plan)
		}
		return nil
	}
	return fmt.Errorf("%s requires sign in", modelName)
}

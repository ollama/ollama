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
	System              string
	Format              string
	Options             map[string]any
	Think               *api.ThinkValue
	KeepAlive           *api.Duration
	ContextWindowTokens int
	AllowAllTools       bool
	ToolsDisabled       bool
	MultiModal          bool
}

func saveLastAgentModel(model string) error {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	return config.SetLastModel(model)
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
	cwd := agentWorkingDir()
	contextWindowForModel := func(ctx context.Context, model string, fallback int) int {
		return agentContextWindowForModel(ctx, client, model, fallback)
	}

	var skillCatalog *coreagent.SkillCatalog
	reloadSkills := func() (*coreagent.SkillCatalog, error) {
		catalog, err := coreagent.LoadDefaultSkills(cwd)
		if err != nil {
			return nil, err
		}
		if ignored := catalog.ExcludeNames(agentchat.BuiltinSlashCommandNames()); len(ignored) > 0 {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m ignoring agent skill(s): %s\n", strings.Join(ignored, ", "))
		}
		for _, diagnostic := range catalog.Diagnostics() {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m ignored invalid agent skill: %v\n", diagnostic)
		}
		skillCatalog = catalog
		return catalog, nil
	}
	if _, err := reloadSkills(); err != nil {
		return fmt.Errorf("load agent skills: %w", err)
	}
	var registry *coreagent.Registry
	registryForModel := func(ctx context.Context, model string) *coreagent.Registry {
		return agentToolsRegistry(ctx, client, model, skillCatalog)
	}
	if opts.Model != "" {
		registry = agentToolsRegistry(cmd.Context(), client, opts.Model, skillCatalog)
	}
	systemPrompt := agentSystemPromptWithWorkingDir(opts.Model, opts.System, agentSkillSystemContext(skillCatalog, registry, opts.ToolsDisabled), cwd)

	_, err := agentchat.Run(cmd.Context(), agentchat.Options{
		Model:                opts.Model,
		Client:               client,
		Tools:                registry,
		ToolRegistryForModel: registryForModel,
		ToolsDisabled:        opts.ToolsDisabled,
		MultiModalForModel: func(ctx context.Context, model string) bool {
			return agentModelSupportsMultimodal(ctx, client, model)
		},
		ModelOptions: func(ctx context.Context) ([]agentchat.ModelOption, error) {
			return agentModelOptions(ctx, client)
		},
		OnModelSelected: func(_ context.Context, model string) error {
			return config.SetLastModel(model)
		},
		SystemPromptForModel: func(ctx context.Context, model string, registry *coreagent.Registry, toolsDisabled bool) string {
			return agentSystemPromptWithWorkingDir(model, agentSystemFromShow(ctx, client, model), agentSkillSystemContext(skillCatalog, registry, toolsDisabled), cwd)
		},
		Skills:              skillCatalog,
		ImportSkills:        coreagent.ImportSkills,
		ReloadSkills:        reloadSkills,
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
			return contextWindowForModel(ctx, model, fallback)
		},
		PreloadModel: func(ctx context.Context, model string, think *api.ThinkValue) (int, error) {
			return preloadAgentModelIfLocal(ctx, client, opts, model, think)
		},
		CheckCloudModel: func(ctx context.Context, model, requiredPlan string) error {
			return ensureCloudModelAccess(ctx, client, model, requiredPlan)
		},
		OpenBrowser: launch.OpenBrowser,
		PollCloudAuth: func(ctx context.Context) (string, bool, error) {
			user, err := client.Whoami(ctx)
			if err != nil {
				return "", false, err
			}
			if user == nil || user.Name == "" {
				return "", false, nil
			}
			return user.Name, true, nil
		},
	})
	return err
}

func agentSkillSystemContext(catalog *coreagent.SkillCatalog, registry *coreagent.Registry, toolsDisabled bool) string {
	if toolsDisabled || registry == nil {
		return ""
	}
	if _, ok := registry.Get("skill"); !ok {
		return ""
	}
	return catalog.SystemContext()
}

func selectAgentModel(ctx context.Context, client *api.Client, current string) (string, error) {
	models, err := agentModelOptions(ctx, client)
	if err != nil {
		return "", err
	}
	if len(models) == 0 {
		return "", errors.New("no models available, run 'ollama pull <model>' first")
	}

	items := agentSelectionItems(models)
	switch {
	case launch.DefaultSingleSelectorWithUpdates != nil:
		return launch.DefaultSingleSelectorWithUpdates("Select model to run:", items, current, nil)
	case launch.DefaultSingleSelector != nil:
		return launch.DefaultSingleSelector("Select model to run:", items, current)
	default:
		return "", errors.New("no selector configured")
	}
}

func agentSelectionItems(models []agentchat.ModelOption) []launch.SelectionItem {
	items := make([]launch.SelectionItem, 0, len(models))
	for _, model := range models {
		items = append(items, launch.SelectionItem{
			Name:              model.Name,
			Description:       agentSelectionDescription(model),
			Recommended:       model.Recommended,
			AvailabilityBadge: model.AvailabilityBadge,
		})
	}
	return items
}

func agentSelectionDescription(model agentchat.ModelOption) string {
	return strings.TrimSpace(model.Description)
}

var agentGetwd = os.Getwd

func agentWorkingDir() string {
	cwd, err := agentGetwd()
	if err != nil {
		return ""
	}
	return cwd
}

func agentSystemPromptWithWorkingDir(modelName string, modelSystem string, extra string, workingDir string) string {
	return agentSystemPromptAtWithWorkingDir(time.Now(), modelName, modelSystem, extra, workingDir)
}

func agentSystemPromptAtWithWorkingDir(now time.Time, modelName string, modelSystem string, extra string, workingDir string) string {
	var parts []string
	parts = append(parts, agentDefaultSystemPromptWithWorkingDir(now, modelName, workingDir))
	if strings.TrimSpace(modelSystem) != "" {
		parts = append(parts, strings.TrimSpace(modelSystem))
	}
	if strings.TrimSpace(extra) != "" {
		parts = append(parts, strings.TrimSpace(extra))
	}
	return strings.Join(parts, "\n\n")
}

func agentDefaultSystemPromptWithWorkingDir(now time.Time, modelName string, workingDir string) string {
	date := now.Format("Monday, January 2, 2006")
	shellName := "bash"
	if runtime.GOOS == "windows" {
		shellName = "PowerShell"
	}
	parts := []string{
		"You are running in Ollama, in a harness to help the user accomplish tasks, and the model is " + modelName + ".",
		"",
		"Current date: " + date + ".",
		"",
	}
	parts = append(parts,
		"Be concise, practical, and action-oriented. Use tools when they materially help. Verify current or fast-changing facts with web tools when available; otherwise state uncertainty.",
		"",
		"Use "+shellName+" carefully. Prefer read-only inspection first. Stay within the current working directory unless explicitly asked. Surface intent before risky actions such as writes, deletes, moves, installs, git state changes, service changes, sudo, secrets access, network scripts, or commands outside the working directory. Request approval when required and do not work around denied approvals.",
		"",
		"Tell the user about meaningful changes, verification, failures, blockers, assumptions, and risks. Summarize routine tool output instead of dumping it.",
	)
	if workingDir != "" {
		parts = append(parts, "Current working directory: "+strconv.Quote(workingDir)+".")
	}
	return strings.Join(parts, "\n")
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

func agentToolsRegistry(ctx context.Context, client *api.Client, modelName string, skillCatalog *coreagent.SkillCatalog) *coreagent.Registry {
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
	if len(skillCatalog.List()) > 0 {
		registry.Register(&agenttools.Skill{Catalog: skillCatalog})
	}

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
	if tokens := loadedContextWindowForModel(ctx, client, modelName); tokens > 0 {
		return tokens
	}
	if modelref.HasExplicitCloudSource(modelName) {
		if tokens := agentRecommendationContextWindowForModel(ctx, client, modelName); tokens > 0 {
			return tokens
		}
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

func agentRecommendationContextWindowForModel(ctx context.Context, client *api.Client, modelName string) int {
	if client == nil {
		return 0
	}
	recs, err := client.ModelRecommendationsExperimental(ctx)
	if err != nil || recs == nil {
		return 0
	}
	return contextWindowFromRecommendations(modelName, recs.Recommendations)
}

func contextWindowFromRecommendations(modelName string, recommendations []api.ModelRecommendation) int {
	for _, rec := range recommendations {
		if rec.ContextLength <= 0 {
			continue
		}
		if sameModelRef(modelName, rec.Model) {
			return rec.ContextLength
		}
	}
	return 0
}

func sameModelRef(a, b string) bool {
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

func showResponseContextWindow(resp *api.ShowResponse) int {
	if resp == nil {
		return 0
	}
	if resp.Details.ContextLength > 0 {
		return resp.Details.ContextLength
	}
	if n, ok := numericModelInfo(resp.ModelInfo["general.context_length"]); ok {
		return n
	}
	best := 0
	for key, value := range resp.ModelInfo {
		if key != "context_length" && !strings.HasSuffix(key, ".context_length") {
			continue
		}
		if n, ok := numericModelInfo(value); ok && n > best {
			best = n
		}
	}
	return best
}

func numericModelInfo(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, v > 0
	case int32:
		return int(v), v > 0
	case int64:
		return int(v), v > 0
	case uint:
		return int(v), v > 0
	case uint32:
		return int(v), v > 0
	case uint64:
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

func preloadAgentModelIfLocal(ctx context.Context, client *api.Client, opts agentTUIOptions, modelName string, think *api.ThinkValue) (int, error) {
	modelName = strings.TrimSpace(modelName)
	if client == nil || modelName == "" {
		return 0, nil
	}
	if modelref.HasExplicitCloudSource(modelName) {
		return 0, nil
	}
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return 0, err
	}
	if info.RemoteHost != "" {
		return 0, nil
	}
	if err := client.Generate(ctx, &api.GenerateRequest{
		Model:     modelName,
		KeepAlive: opts.KeepAlive,
		Options:   opts.Options,
		Think:     think,
	}, func(api.GenerateResponse) error {
		return nil
	}); err != nil {
		return 0, err
	}
	return loadedContextWindowForModel(ctx, client, modelName), nil
}

func loadedContextWindowForModel(ctx context.Context, client *api.Client, modelName string) int {
	if client == nil || strings.TrimSpace(modelName) == "" {
		return 0
	}
	resp, err := client.ListRunning(ctx)
	if err != nil {
		return 0
	}
	return processContextWindowForModel(modelName, resp)
}

func processContextWindowForModel(modelName string, resp *api.ProcessResponse) int {
	if resp == nil {
		return 0
	}
	for _, running := range resp.Models {
		if running.ContextLength <= 0 {
			continue
		}
		if sameModelRef(modelName, running.Name) || sameModelRef(modelName, running.Model) {
			return running.ContextLength
		}
	}
	return 0
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
		if errors.As(err, &authErr) && (authErr.StatusCode == http.StatusUnauthorized || authErr.SigninURL != "") {
			if authErr.SigninURL != "" {
				signInURL = authErr.SigninURL
			}
		} else {
			return badges, signInURLs
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

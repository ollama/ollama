package cmd

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/spf13/cobra"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/chatstore"
	"github.com/ollama/ollama/agent/skills"
	agenttools "github.com/ollama/ollama/agent/tools"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/tui"
	"github.com/ollama/ollama/format"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
)

type AgentTUIOptions struct {
	Model               string
	Prompt              string
	LoadedMessages      []api.Message
	Messages            []api.Message
	Images              []api.ImageData
	WordWrap            bool
	Format              string
	Options             map[string]any
	Think               *api.ThinkValue
	HideThinking        bool
	KeepAlive           *api.Duration
	ContextWindowTokens int
	Resume              bool
	AutoApproveTools    bool
	Verbose             bool
	MultiModal          bool
	Skill               string
	Skills              *skills.Catalog
}

func agentOptionsFromRunOptions(opts runOptions) AgentTUIOptions {
	return AgentTUIOptions{
		Model:               opts.Model,
		Prompt:              opts.Prompt,
		LoadedMessages:      opts.LoadedMessages,
		Messages:            opts.Messages,
		Images:              opts.Images,
		WordWrap:            opts.WordWrap,
		Format:              opts.Format,
		Options:             opts.Options,
		Think:               opts.Think,
		HideThinking:        opts.HideThinking,
		KeepAlive:           opts.KeepAlive,
		ContextWindowTokens: opts.ContextWindowTokens,
		Resume:              opts.Resume,
		AutoApproveTools:    opts.AutoApproveTools,
		Verbose:             opts.Verbose,
		MultiModal:          opts.MultiModal,
		Skill:               opts.Skill,
	}
}

func GenerateAgentTUI(cmd *cobra.Command, opts AgentTUIOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}

	var persistentStore *chatstore.Store
	if store, err := chatstore.New(""); err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m chat persistence unavailable: %v\n", err)
	} else {
		persistentStore = store
		defer persistentStore.Close()
	}

	newChatID := func(ctx context.Context) (string, error) {
		u, err := uuid.NewV7()
		if err != nil {
			u = uuid.Must(uuid.NewRandom())
		}
		chatID := u.String()
		if persistentStore != nil {
			if err := persistentStore.EnsureChat(ctx, chatID, ""); err != nil {
				return "", err
			}
		}
		return chatID, nil
	}

	chatID := ""
	var resumedMessages []api.Message
	if opts.Resume {
		if persistentStore == nil {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m chat resume unavailable: persistence is disabled\n")
		} else if chat, err := persistentStore.LatestChatForModel(cmd.Context(), opts.Model); err == nil {
			chatID = chat.ID
			resumedMessages = chat.Messages
		} else if errors.Is(err, sql.ErrNoRows) {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m no saved chat for %s; starting a new chat\n", opts.Model)
		} else {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not resume chat: %v\n", err)
		}
	}
	if chatID == "" {
		var err error
		chatID, err = newChatID(cmd.Context())
		if err != nil {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not create persistent chat: %v\n", err)
			persistentStore = nil
			chatID, _ = newChatID(cmd.Context())
		}
	}

	skillCatalog := opts.Skills
	if skillCatalog == nil {
		skillCatalog = loadAgentSkills()
	}

	registry := agentToolsRegistry(cmd.Context(), client, opts.Model, skillCatalog)
	opts.ContextWindowTokens = contextWindowTokensForRun(cmd.Context(), client, opts.Model, opts.ContextWindowTokens)
	approval := coreagent.ApprovalHandler(coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{}))
	if opts.AutoApproveTools {
		approval = coreagent.AutoAllowApproval{}
	}

	messages := slices.Clone(opts.LoadedMessages)
	messages = append(messages, resumedMessages...)
	messages = append(messages, opts.Messages...)

	_, err = tui.RunAgentChat(cmd.Context(), tui.ChatOptions{
		Model:    opts.Model,
		ChatID:   chatID,
		Messages: messages,
		Client:   client,
		Store:    persistentStore,
		Tools:    registry,
		ToolRegistryForModel: func(ctx context.Context, model string) *coreagent.Registry {
			return agentToolsRegistry(ctx, client, model, skillCatalog)
		},
		ModelOptions: func(ctx context.Context) ([]tui.ChatModelOption, error) {
			return agentModelOptions(ctx, client)
		},
		OnModelSelected: func(_ context.Context, model string) error {
			return config.SetLastModel(model)
		},
		SystemPromptForModel: func(_ context.Context, _ string, registry *coreagent.Registry) string {
			return agentSystemPrompt(skillCatalog, registry != nil && registry.Has("skill"), "")
		},
		Approval:         approval,
		AutoApproveTools: opts.AutoApproveTools,
		Skills:           skillCatalog,
		SystemPrompt:     agentSystemPrompt(skillCatalog, registry != nil && registry.Has("skill"), ""),
		WorkingDir:       cwd,
		Format:           opts.Format,
		Options:          opts.Options,
		Think:            opts.Think,
		KeepAlive:        opts.KeepAlive,
		HideThinking:     opts.HideThinking,
		Verbose:          opts.Verbose,
		Compactor: coreagent.NewSimpleCompactor(client, persistentStore, coreagent.CompactionOptions{
			ContextWindowTokens: opts.ContextWindowTokens,
		}),
		ContextWindowTokens: opts.ContextWindowTokens,
		ContextWindowTokensForModel: func(ctx context.Context, model string, fallback int) int {
			return contextWindowTokensForRun(ctx, client, model, fallback)
		},
		NewChat: newChatID,
	})
	return err
}

func GenerateAgentHeadless(cmd *cobra.Command, opts AgentTUIOptions) error {
	if strings.TrimSpace(opts.Prompt) == "" {
		return errors.New("agent headless mode requires a prompt or stdin")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}

	var persistentStore *chatstore.Store
	if store, err := chatstore.New(""); err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m chat persistence unavailable: %v\n", err)
	} else {
		persistentStore = store
		defer persistentStore.Close()
	}

	newChatID := func(ctx context.Context) (string, error) {
		u, err := uuid.NewV7()
		if err != nil {
			u = uuid.Must(uuid.NewRandom())
		}
		chatID := u.String()
		if persistentStore != nil {
			if err := persistentStore.EnsureChat(ctx, chatID, ""); err != nil {
				return "", err
			}
		}
		return chatID, nil
	}

	chatID := ""
	var resumedMessages []api.Message
	if opts.Resume {
		if persistentStore == nil {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m chat resume unavailable: persistence is disabled\n")
		} else {
			var chat *chatstore.Chat
			var err error
			if opts.Model == "" {
				chat, err = persistentStore.LatestChat(cmd.Context())
			} else {
				chat, err = persistentStore.LatestChatForModel(cmd.Context(), opts.Model)
			}
			if err == nil {
				chatID = chat.ID
				if opts.Model == "" {
					opts.Model = chat.Model
				}
				resumedMessages = chat.Messages
			} else if errors.Is(err, sql.ErrNoRows) && opts.Model != "" {
				fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m no saved chat for %s; starting a new chat\n", opts.Model)
			} else if errors.Is(err, sql.ErrNoRows) {
				return errors.New("no saved chat to resume; pass a model to start a new chat")
			} else {
				return fmt.Errorf("could not resume chat: %w", err)
			}
		}
	}
	if opts.Model == "" {
		return errors.New("model is required")
	}
	if chatID == "" {
		chatID, err = newChatID(cmd.Context())
		if err != nil {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not create persistent chat: %v\n", err)
			persistentStore = nil
			chatID, _ = newChatID(cmd.Context())
		}
	}

	skillCatalog := opts.Skills
	if skillCatalog == nil {
		skillCatalog = loadAgentSkills()
	}

	registry := agentToolsRegistry(cmd.Context(), client, opts.Model, skillCatalog)
	opts.ContextWindowTokens = contextWindowTokensForRun(cmd.Context(), client, opts.Model, opts.ContextWindowTokens)
	approval := coreagent.ApprovalHandler(coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{}))
	if opts.AutoApproveTools {
		approval = coreagent.AutoAllowApproval{}
	}
	messages := slices.Clone(opts.LoadedMessages)
	messages = append(messages, resumedMessages...)
	messages = append(messages, opts.Messages...)

	prompt := opts.Prompt
	images := slices.Clone(opts.Images)
	if opts.MultiModal {
		var err error
		prompt, images, err = extractFileData(prompt)
		if err != nil {
			return err
		}
	}
	systemPrompt := agentSystemPrompt(skillCatalog, registry != nil && registry.Has("skill"), "")
	if strings.TrimSpace(opts.Skill) == "" {
		if skill, request, ok := skillFromPrompt(skillCatalog, prompt); ok {
			opts.Skill = skill.Name
			prompt = request
		}
	}
	if strings.TrimSpace(opts.Skill) != "" {
		skill, ok := skillCatalog.Find(opts.Skill)
		if !ok {
			return fmt.Errorf("unknown skill: %s", opts.Skill)
		}
		manualPrompt, err := skills.ManualSystemPrompt(skill)
		if err != nil {
			return err
		}
		systemPrompt = agentSystemPrompt(skillCatalog, registry != nil && registry.Has("skill"), manualPrompt)
		prompt = skills.ManualUserPrompt(skill.Name, prompt)
	}

	runCtx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)
	defer signal.Stop(sigChan)
	go func() {
		select {
		case <-sigChan:
			cancel()
		case <-runCtx.Done():
		}
	}()

	sink := &agentHeadlessEventSink{}
	session := &coreagent.Session{
		Client:     client,
		Store:      persistentStore,
		Events:     sink,
		Tools:      registry,
		Approval:   approval,
		WorkingDir: cwd,
		Compactor: coreagent.NewSimpleCompactor(client, persistentStore, coreagent.CompactionOptions{
			ContextWindowTokens: opts.ContextWindowTokens,
		}),
	}
	result, err := session.Run(runCtx, coreagent.RunOptions{
		ChatID:       chatID,
		Model:        opts.Model,
		SystemPrompt: systemPrompt,
		Messages:     messages,
		NewMessages:  []api.Message{{Role: "user", Content: prompt, Images: images}},
		Format:       opts.Format,
		Options:      opts.Options,
		Think:        opts.Think,
		KeepAlive:    opts.KeepAlive,
		UseTools:     registry != nil,
	})
	if err != nil {
		return err
	}
	if sink.wroteContent {
		fmt.Fprintln(os.Stdout)
	}

	verbose := opts.Verbose
	if cmd != nil && cmd.Flags().Lookup("verbose") != nil {
		flagVerbose, err := cmd.Flags().GetBool("verbose")
		if err != nil {
			return err
		}
		verbose = verbose || flagVerbose
	}
	if verbose {
		result.Latest.Summary()
	}
	return nil
}

func skillFromPrompt(catalog *skills.Catalog, prompt string) (skills.Skill, string, bool) {
	if catalog == nil || catalog.Empty() {
		return skills.Skill{}, "", false
	}
	prompt = strings.TrimSpace(prompt)
	if !strings.HasPrefix(prompt, "/") {
		return skills.Skill{}, "", false
	}
	command, rest, _ := strings.Cut(prompt, " ")
	skill, ok := catalog.Find(command)
	return skill, strings.TrimSpace(rest), ok
}

type agentHeadlessEventSink struct {
	wroteContent bool
}

func (s *agentHeadlessEventSink) Emit(event coreagent.Event) error {
	switch event.Type {
	case coreagent.EventMessageDelta:
		if event.Content != "" {
			fmt.Fprint(os.Stdout, event.Content)
			s.wroteContent = true
		}
	case coreagent.EventToolStarted:
		fmt.Fprintf(os.Stderr, "• %s in progress\n", agentHeadlessToolLabel(event.ToolName, event.Args))
	case coreagent.EventToolFinished:
		status := "done"
		if event.Error != "" {
			status = "failed"
		}
		fmt.Fprintf(os.Stderr, "• %s %s\n", agentHeadlessToolLabel(event.ToolName, event.Args), status)
	case coreagent.EventToolsUnavailable:
		fmt.Fprintln(os.Stderr, "Tools are unavailable for this model.")
	case coreagent.EventCompactionSkipped:
		if event.Content != "" {
			fmt.Fprintf(os.Stderr, "%s\n", event.Content)
		}
	case coreagent.EventError:
		if event.Error != "" {
			fmt.Fprintf(os.Stderr, "error: %s\n", event.Error)
		}
	}
	return nil
}

func agentHeadlessToolLabel(name string, args map[string]any) string {
	displayName := name
	switch name {
	case "web_search":
		displayName = "Web Search"
	case "web_fetch":
		displayName = "Web Fetch"
	case "bash":
		displayName = "Bash"
	case "read":
		displayName = "Read"
	case "edit":
		displayName = "Edit"
	}

	for _, key := range []string{"query", "url", "command", "path"} {
		if value, ok := args[key].(string); ok && strings.TrimSpace(value) != "" {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(value))
		}
	}
	return displayName
}

func loadAgentSkills() *skills.Catalog {
	catalog, err := skills.LoadDefault()
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not load skills: %v\n", err)
		return nil
	}
	for _, warning := range catalog.Warnings {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m skill ignored: %s\n", warning)
	}
	return catalog
}

func agentSystemPrompt(catalog *skills.Catalog, skillToolAvailable bool, extra string) string {
	return agentSystemPromptAt(time.Now(), catalog, skillToolAvailable, extra)
}

func agentSystemPromptAt(now time.Time, catalog *skills.Catalog, skillToolAvailable bool, extra string) string {
	var parts []string
	parts = append(parts, agentDefaultSystemPrompt(now))
	if catalogPrompt := catalog.SystemPrompt(skillToolAvailable); strings.TrimSpace(catalogPrompt) != "" {
		parts = append(parts, catalogPrompt)
	}
	if strings.TrimSpace(extra) != "" {
		parts = append(parts, strings.TrimSpace(extra))
	}
	return strings.Join(parts, "\n\n")
}

func agentDefaultSystemPrompt(now time.Time) string {
	date := now.Format("Monday, January 2, 2006")
	return strings.Join([]string{
		"You are Ollama, a helpful local AI assistant.",
		"",
		"Current date: " + date + ".",
		"",
		"Be concise, practical, and action-oriented. Use tools when they materially help. Verify current or fast-changing facts with web tools when available; otherwise state uncertainty.",
		"",
		"Use bash carefully. Prefer read-only inspection first. Stay within the current working directory unless explicitly asked. Surface intent before risky actions such as writes, deletes, moves, installs, git state changes, service changes, sudo, secrets access, network scripts, or commands outside the working directory. Request approval when required and do not work around denied approvals.",
		"",
		"Tell the user about meaningful changes, verification, failures, blockers, assumptions, and risks. Summarize routine tool output instead of dumping it.",
	}, "\n")
}

func agentModelOptions(ctx context.Context, client *api.Client) ([]tui.ChatModelOption, error) {
	if client == nil {
		return nil, errors.New("model picker requires an API client")
	}

	list, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	seen := make(map[string]struct{})
	var options []tui.ChatModelOption
	add := func(name, description string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		key := strings.ToLower(name)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		options = append(options, tui.ChatModelOption{Name: name, Description: strings.TrimSpace(description)})
	}

	if disabled, known := agentCloudStatusDisabled(ctx, client); !known || !disabled {
		if recs, err := client.ModelRecommendationsExperimental(ctx); err == nil {
			for _, rec := range recs.Recommendations {
				name := strings.TrimSpace(rec.Model)
				if !modelref.HasExplicitCloudSource(name) {
					continue
				}
				add(name, agentRecommendationDescription(rec))
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
		add(name, agentLocalModelDescription(model))
	}

	return options, nil
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
	if rec.RequiredPlan != "" {
		parts = append(parts, rec.RequiredPlan+" plan")
	}
	return strings.Join(parts, " · ")
}

func agentLocalModelDescription(model api.ListModelResponse) string {
	parts := []string{"local"}
	if model.Details.Family != "" {
		parts = append(parts, model.Details.Family)
	}
	if model.Details.ParameterSize != "" {
		parts = append(parts, model.Details.ParameterSize)
	}
	if model.Details.QuantizationLevel != "" {
		parts = append(parts, model.Details.QuantizationLevel)
	}
	if model.Details.ContextLength > 0 {
		parts = append(parts, format.HumanNumber(uint64(model.Details.ContextLength))+" ctx")
	}
	if model.Size > 0 {
		parts = append(parts, format.HumanBytes(model.Size))
	}
	return strings.Join(parts, " · ")
}

func agentToolsRegistry(ctx context.Context, client *api.Client, modelName string, catalog *skills.Catalog) *coreagent.Registry {
	supportsTools, err := agentModelSupportsTools(ctx, client, modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not check model capabilities: %v\n", err)
	}
	if !supportsTools {
		return nil
	}

	registry := coreagent.NewRegistry()
	if os.Getenv("OLLAMA_AGENT_DISABLE_BASH") == "" {
		registry.Register(agenttools.NewBash())
	}
	registry.Register(agenttools.NewRead())
	registry.Register(agenttools.NewEdit())
	if !catalog.Empty() {
		registry.Register(agenttools.NewSkill(catalog))
	}

	if os.Getenv("OLLAMA_AGENT_DISABLE_WEBSEARCH") == "" {
		if disabled, known := agentCloudStatusDisabled(ctx, client); !known || !disabled {
			registry.Register(agenttools.NewWebSearch())
			registry.Register(agenttools.NewWebFetch())
		} else {
			fmt.Fprintf(os.Stderr, "%s\n", internalcloud.DisabledError("web search is unavailable"))
		}
	}
	return registry
}

func agentModelSupportsTools(ctx context.Context, client *api.Client, modelName string) (bool, error) {
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return false, err
	}

	return slices.Contains(resp.Capabilities, model.CapabilityTools), nil
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

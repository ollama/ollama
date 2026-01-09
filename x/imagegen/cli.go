// cli.go provides CLI commands for image generation models.
//
// TODO (jmorganca): Integrate these commands into cmd/cmd.go when stable.
// Currently these are separate to keep experimental code isolated.

package imagegen

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
)

// ImageGenOptions holds options for image generation.
// These can be set via environment variables or interactive commands.
type ImageGenOptions struct {
	Width          int
	Height         int
	Steps          int
	Seed           int
	NegativePrompt string
}

// DefaultOptions returns the default image generation options.
func DefaultOptions() ImageGenOptions {
	return ImageGenOptions{
		Width:  1024,
		Height: 1024,
		Steps:  9,
		Seed:   0, // 0 means random
	}
}

// Show displays information about an image generation model.
func Show(modelName string, verbose bool, w io.Writer) error {
	manifest, err := LoadManifest(modelName)
	if err != nil {
		return fmt.Errorf("failed to load manifest: %w", err)
	}

	// Count tensors and total size per component
	type tensorInfo struct {
		name  string
		dtype string
		shape []int32
		size  int64
	}
	type componentInfo struct {
		tensors []tensorInfo
		size    int64
	}
	components := make(map[string]*componentInfo)
	var totalSize int64

	for _, layer := range manifest.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.tensor" {
			if idx := strings.Index(layer.Name, "/"); idx > 0 {
				comp := layer.Name[:idx]
				tensorName := layer.Name[idx+1:]
				if components[comp] == nil {
					components[comp] = &componentInfo{}
				}
				components[comp].tensors = append(components[comp].tensors, tensorInfo{
					name:  tensorName,
					dtype: layer.Dtype,
					shape: layer.Shape,
					size:  layer.Size,
				})
				components[comp].size += layer.Size
			}
			totalSize += layer.Size
		}
	}

	// Read model_index.json for architecture
	var architecture string
	if data, err := manifest.ReadConfig("model_index.json"); err == nil {
		var index struct {
			Architecture string `json:"architecture"`
		}
		if json.Unmarshal(data, &index) == nil {
			architecture = index.Architecture
		}
	}

	// Read component configs as raw maps for flexible display
	componentConfigs := make(map[string]map[string]any)
	for _, comp := range []string{"transformer", "text_encoder", "vae", "scheduler"} {
		configPath := comp + "/config.json"
		if comp == "scheduler" {
			configPath = "scheduler/scheduler_config.json"
		}
		if data, err := manifest.ReadConfig(configPath); err == nil {
			var cfg map[string]any
			if json.Unmarshal(data, &cfg) == nil {
				componentConfigs[comp] = cfg
			}
		}
	}

	// Extract key values for summary display
	var transformerDim, transformerLayers, transformerHeads int
	var latentChannels int
	if cfg, ok := componentConfigs["transformer"]; ok {
		if v, ok := cfg["dim"].(float64); ok {
			transformerDim = int(v)
		}
		if v, ok := cfg["n_layers"].(float64); ok {
			transformerLayers = int(v)
		}
		if v, ok := cfg["n_heads"].(float64); ok {
			transformerHeads = int(v)
		}
		if v, ok := cfg["in_channels"].(float64); ok {
			latentChannels = int(v)
		}
	}

	// Get requires version from config
	var requires string
	configLayer := manifest.Manifest.Config
	if configLayer.Digest != "" {
		if data, err := os.ReadFile(manifest.BlobPath(configLayer.Digest)); err == nil {
			var cfg struct {
				Requires string `json:"requires,omitempty"`
			}
			if json.Unmarshal(data, &cfg) == nil {
				requires = cfg.Requires
			}
		}
	}

	// Estimate parameter count from total size (assuming BF16 = 2 bytes per param)
	paramCount := totalSize / 2
	paramStr := formatParamCount(paramCount)

	// Print Model info
	fmt.Fprintln(w, "  Model")
	if architecture != "" {
		fmt.Fprintf(w, "    %-20s %s\n", "architecture", architecture)
	}
	fmt.Fprintf(w, "    %-20s %s\n", "parameters", paramStr)
	if transformerDim > 0 {
		fmt.Fprintf(w, "    %-20s %d\n", "hidden size", transformerDim)
	}
	if transformerLayers > 0 {
		fmt.Fprintf(w, "    %-20s %d\n", "layers", transformerLayers)
	}
	if transformerHeads > 0 {
		fmt.Fprintf(w, "    %-20s %d\n", "attention heads", transformerHeads)
	}
	fmt.Fprintf(w, "    %-20s %s\n", "quantization", "BF16")
	if requires != "" {
		fmt.Fprintf(w, "    %-20s %s\n", "requires", requires)
	}
	fmt.Fprintln(w)

	// Print Capabilities
	fmt.Fprintln(w, "  Capabilities")
	fmt.Fprintf(w, "    %s\n", "image")
	fmt.Fprintln(w)

	// Print Parameters (generation defaults)
	fmt.Fprintln(w, "  Parameters")
	fmt.Fprintf(w, "    %-20s %d\n", "default steps", 9)
	fmt.Fprintf(w, "    %-20s %dx%d\n", "default size", 1024, 1024)
	if cfg, ok := componentConfigs["scheduler"]; ok {
		if v, ok := cfg["num_train_timesteps"].(float64); ok && v > 0 {
			fmt.Fprintf(w, "    %-20s %d\n", "train timesteps", int(v))
		}
	}
	if cfg, ok := componentConfigs["transformer"]; ok {
		if ps, ok := cfg["patch_size"].([]any); ok && len(ps) > 0 {
			if v, ok := ps[0].(float64); ok {
				fmt.Fprintf(w, "    %-20s %d\n", "patch size", int(v))
			}
		}
	}
	if latentChannels > 0 {
		fmt.Fprintf(w, "    %-20s %d\n", "latent channels", latentChannels)
	}
	fmt.Fprintln(w)

	// Print Components (with config and tensors in verbose mode)
	fmt.Fprintln(w, "  Components")
	for _, name := range []string{"text_encoder", "transformer", "vae", "scheduler"} {
		info := components[name]
		cfg := componentConfigs[name]

		// Skip if no info and no config
		if info == nil && cfg == nil {
			continue
		}

		if verbose {
			// Header with tensor summary
			if info != nil {
				fmt.Fprintf(w, "    %s (%d tensors, %s)\n", name, len(info.tensors), formatBytes(info.size))
			} else {
				fmt.Fprintf(w, "    %s\n", name)
			}

			// Config values
			if cfg != nil {
				fmt.Fprintln(w, "      config:")
				printConfigMap(w, cfg, "        ")
			}

			// Tensors
			if info != nil && len(info.tensors) > 0 {
				fmt.Fprintln(w, "      tensors:")
				for _, t := range info.tensors {
					shapeStr := formatShape(t.shape)
					fmt.Fprintf(w, "        %-56s %-6s %s\n", t.name, t.dtype, shapeStr)
				}
			}
			fmt.Fprintln(w)
		} else {
			if info != nil {
				fmt.Fprintf(w, "    %-20s %d tensors, %s\n", name, len(info.tensors), formatBytes(info.size))
			}
		}
	}
	if !verbose {
		fmt.Fprintln(w)
	}

	return nil
}

// formatParamCount formats parameter count as human-readable string.
func formatParamCount(count int64) string {
	if count >= 1_000_000_000 {
		return fmt.Sprintf("%.1fB", float64(count)/1_000_000_000)
	}
	if count >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(count)/1_000_000)
	}
	return fmt.Sprintf("%d", count)
}

// formatShape formats a tensor shape as a string.
func formatShape(shape []int32) string {
	if len(shape) == 0 {
		return "[]"
	}
	parts := make([]string, len(shape))
	for i, v := range shape {
		parts[i] = strconv.Itoa(int(v))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// formatBytes formats bytes as human-readable string.
func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// printConfigMap prints a config map with proper formatting.
func printConfigMap(w io.Writer, cfg map[string]any, indent string) {
	// Sort keys for consistent output, skip internal diffusers keys
	keys := make([]string, 0, len(cfg))
	for k := range cfg {
		if strings.HasPrefix(k, "_diffusers") {
			continue
		}
		keys = append(keys, k)
	}
	// Simple alphabetical sort
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if keys[i] > keys[j] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}

	for _, k := range keys {
		v := cfg[k]
		switch val := v.(type) {
		case float64:
			// Check if it's an integer value
			if val == float64(int64(val)) {
				fmt.Fprintf(w, "%s%-24s %d\n", indent, k, int64(val))
			} else {
				fmt.Fprintf(w, "%s%-24s %g\n", indent, k, val)
			}
		case bool:
			fmt.Fprintf(w, "%s%-24s %v\n", indent, k, val)
		case string:
			fmt.Fprintf(w, "%s%-24s %s\n", indent, k, val)
		case []any:
			// Format arrays compactly
			fmt.Fprintf(w, "%s%-24s %v\n", indent, k, val)
		default:
			fmt.Fprintf(w, "%s%-24s %v\n", indent, k, v)
		}
	}
}

// RegisterFlags adds image generation flags to the given command.
// Flags are hidden since they only apply to image generation models.
func RegisterFlags(cmd *cobra.Command) {
	cmd.Flags().Int("width", 1024, "Image width")
	cmd.Flags().Int("height", 1024, "Image height")
	cmd.Flags().Int("steps", 9, "Denoising steps")
	cmd.Flags().Int("seed", 0, "Random seed (0 for random)")
	cmd.Flags().String("negative", "", "Negative prompt")
	cmd.Flags().MarkHidden("width")
	cmd.Flags().MarkHidden("height")
	cmd.Flags().MarkHidden("steps")
	cmd.Flags().MarkHidden("seed")
	cmd.Flags().MarkHidden("negative")
}

// RunCLI handles the CLI for image generation models.
// Returns true if it handled the request, false if the caller should continue with normal flow.
// Supports flags: --width, --height, --steps, --seed, --negative
func RunCLI(cmd *cobra.Command, name string, prompt string, interactive bool, keepAlive *api.Duration) error {
	// Verify it's a valid image gen model
	if ResolveModelName(name) == "" {
		return fmt.Errorf("unknown image generation model: %s", name)
	}

	// Get options from flags (with env var defaults)
	opts := DefaultOptions()
	if cmd != nil && cmd.Flags() != nil {
		if v, err := cmd.Flags().GetInt("width"); err == nil && v > 0 {
			opts.Width = v
		}
		if v, err := cmd.Flags().GetInt("height"); err == nil && v > 0 {
			opts.Height = v
		}
		if v, err := cmd.Flags().GetInt("steps"); err == nil && v > 0 {
			opts.Steps = v
		}
		if v, err := cmd.Flags().GetInt("seed"); err == nil && v != 0 {
			opts.Seed = v
		}
		if v, err := cmd.Flags().GetString("negative"); err == nil && v != "" {
			opts.NegativePrompt = v
		}
	}

	if interactive {
		return runInteractive(cmd, name, keepAlive, opts)
	}

	// One-shot generation
	return generateImageWithOptions(cmd, name, prompt, keepAlive, opts)
}

// generateImageWithOptions generates an image with the given options.
func generateImageWithOptions(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration, opts ImageGenOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Build request with image gen options encoded in Options fields
	// NumCtx=width, NumGPU=height, NumPredict=steps, Seed=seed
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Options: map[string]any{
			"num_ctx":     opts.Width,
			"num_gpu":     opts.Height,
			"num_predict": opts.Steps,
			"seed":        opts.Seed,
		},
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}

	// Show loading spinner until generation starts
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var stepBar *progress.StepBar
	var imagePath string

	err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
		content := resp.Response

		// Handle progress updates - parse step info and switch to step bar
		if strings.HasPrefix(content, "\rGenerating:") {
			var step, total int
			fmt.Sscanf(content, "\rGenerating: step %d/%d", &step, &total)
			if stepBar == nil && total > 0 {
				spinner.Stop()
				stepBar = progress.NewStepBar("Generating", total)
				p.Add("", stepBar)
			}
			if stepBar != nil {
				stepBar.Set(step)
			}
			return nil
		}

		// Handle final response with image path
		if resp.Done && strings.Contains(content, "Image saved to:") {
			if idx := strings.Index(content, "Image saved to: "); idx >= 0 {
				imagePath = strings.TrimSpace(content[idx+16:])
			}
		}

		return nil
	})

	p.Stop()
	if err != nil {
		return err
	}

	if imagePath != "" {
		displayImageInTerminal(imagePath)
		fmt.Printf("Image saved to: %s\n", imagePath)
	}

	return nil
}

// runInteractive runs an interactive REPL for image generation.
func runInteractive(cmd *cobra.Command, modelName string, keepAlive *api.Duration, opts ImageGenOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:      ">>> ",
		Placeholder: "Describe an image to generate (/help for commands)",
	})
	if err != nil {
		return err
	}

	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}
			continue
		case err != nil:
			return err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Handle commands
		switch {
		case strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/?"), strings.HasPrefix(line, "/help"):
			printInteractiveHelp(opts)
			continue
		case strings.HasPrefix(line, "/set "):
			if err := handleSetCommand(line[5:], &opts); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		case strings.HasPrefix(line, "/show"):
			printCurrentSettings(opts)
			continue
		case strings.HasPrefix(line, "/"):
			fmt.Fprintf(os.Stderr, "Unknown command: %s (try /help)\n", line)
			continue
		}

		// Generate image with current options
		req := &api.GenerateRequest{
			Model:  modelName,
			Prompt: line,
			Options: map[string]any{
				"num_ctx":     opts.Width,
				"num_gpu":     opts.Height,
				"num_predict": opts.Steps,
				"seed":        opts.Seed,
			},
		}
		if keepAlive != nil {
			req.KeepAlive = keepAlive
		}

		// Show loading spinner until generation starts
		p := progress.NewProgress(os.Stderr)
		spinner := progress.NewSpinner("")
		p.Add("", spinner)

		var stepBar *progress.StepBar
		var imagePath string

		err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
			content := resp.Response

			// Handle progress updates - parse step info and switch to step bar
			if strings.HasPrefix(content, "\rGenerating:") {
				var step, total int
				fmt.Sscanf(content, "\rGenerating: step %d/%d", &step, &total)
				if stepBar == nil && total > 0 {
					spinner.Stop()
					stepBar = progress.NewStepBar("Generating", total)
					p.Add("", stepBar)
				}
				if stepBar != nil {
					stepBar.Set(step)
				}
				return nil
			}

			// Handle final response with image path
			if resp.Done && strings.Contains(content, "Image saved to:") {
				if idx := strings.Index(content, "Image saved to: "); idx >= 0 {
					imagePath = strings.TrimSpace(content[idx+16:])
				}
			}

			return nil
		})

		p.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		// Copy image to current directory with descriptive name
		if imagePath != "" {
			// Create filename from prompt (sanitized)
			safeName := sanitizeFilename(line)
			if len(safeName) > 50 {
				safeName = safeName[:50]
			}
			timestamp := time.Now().Format("20060102-150405")
			newName := fmt.Sprintf("%s-%s.png", safeName, timestamp)

			// Copy file to CWD
			if err := copyFile(imagePath, newName); err != nil {
				fmt.Fprintf(os.Stderr, "Error saving to current directory: %v\n", err)
				displayImageInTerminal(imagePath)
				fmt.Printf("Image saved to: %s\n", imagePath)
			} else {
				displayImageInTerminal(newName)
				fmt.Printf("Image saved to: %s\n", newName)
			}
		}

		fmt.Println()
	}
}

// sanitizeFilename removes characters that aren't safe for filenames.
func sanitizeFilename(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, " ", "-")
	// Remove any character that's not alphanumeric or hyphen
	var result strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			result.WriteRune(r)
		}
	}
	return result.String()
}

// copyFile copies a file from src to dst.
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// printInteractiveHelp prints help for interactive mode commands.
func printInteractiveHelp(opts ImageGenOptions) {
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  /set width <n>     Set image width (current:", opts.Width, ")")
	fmt.Fprintln(os.Stderr, "  /set height <n>    Set image height (current:", opts.Height, ")")
	fmt.Fprintln(os.Stderr, "  /set steps <n>     Set denoising steps (current:", opts.Steps, ")")
	fmt.Fprintln(os.Stderr, "  /set seed <n>      Set random seed (current:", opts.Seed, ", 0=random)")
	fmt.Fprintln(os.Stderr, "  /set negative <s>  Set negative prompt")
	fmt.Fprintln(os.Stderr, "  /show              Show current settings")
	fmt.Fprintln(os.Stderr, "  /bye               Exit")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Or type a prompt to generate an image.")
	fmt.Fprintln(os.Stderr)
}

// printCurrentSettings prints the current image generation settings.
func printCurrentSettings(opts ImageGenOptions) {
	fmt.Fprintf(os.Stderr, "Current settings:\n")
	fmt.Fprintf(os.Stderr, "  width:    %d\n", opts.Width)
	fmt.Fprintf(os.Stderr, "  height:   %d\n", opts.Height)
	fmt.Fprintf(os.Stderr, "  steps:    %d\n", opts.Steps)
	fmt.Fprintf(os.Stderr, "  seed:     %d (0=random)\n", opts.Seed)
	if opts.NegativePrompt != "" {
		fmt.Fprintf(os.Stderr, "  negative: %s\n", opts.NegativePrompt)
	}
	fmt.Fprintln(os.Stderr)
}

// handleSetCommand handles /set commands to change options.
func handleSetCommand(args string, opts *ImageGenOptions) error {
	parts := strings.SplitN(args, " ", 2)
	if len(parts) < 2 {
		return fmt.Errorf("usage: /set <option> <value>")
	}

	key := strings.ToLower(parts[0])
	value := strings.TrimSpace(parts[1])

	switch key {
	case "width", "w":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("width must be a positive integer")
		}
		opts.Width = v
		fmt.Fprintf(os.Stderr, "Set width to %d\n", v)
	case "height", "h":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("height must be a positive integer")
		}
		opts.Height = v
		fmt.Fprintf(os.Stderr, "Set height to %d\n", v)
	case "steps", "s":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("steps must be a positive integer")
		}
		opts.Steps = v
		fmt.Fprintf(os.Stderr, "Set steps to %d\n", v)
	case "seed":
		v, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("seed must be an integer")
		}
		opts.Seed = v
		fmt.Fprintf(os.Stderr, "Set seed to %d\n", v)
	case "negative", "neg", "n":
		opts.NegativePrompt = value
		if value == "" {
			fmt.Fprintln(os.Stderr, "Cleared negative prompt")
		} else {
			fmt.Fprintf(os.Stderr, "Set negative prompt to: %s\n", value)
		}
	default:
		return fmt.Errorf("unknown option: %s (try /help)", key)
	}
	return nil
}

// displayImageInTerminal attempts to render an image inline in the terminal.
// Supports iTerm2, Kitty, WezTerm, Ghostty, and other terminals with inline image support.
// Returns true if the image was displayed, false otherwise.
func displayImageInTerminal(imagePath string) bool {
	// Check if terminal supports inline images
	termProgram := os.Getenv("TERM_PROGRAM")
	kittyWindowID := os.Getenv("KITTY_WINDOW_ID")
	weztermPane := os.Getenv("WEZTERM_PANE")
	ghostty := os.Getenv("GHOSTTY_RESOURCES_DIR")

	// Read the image file
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return false
	}

	encoded := base64.StdEncoding.EncodeToString(data)

	switch {
	case termProgram == "iTerm.app" || termProgram == "WezTerm" || weztermPane != "":
		// iTerm2/WezTerm inline image protocol
		// ESC ] 1337 ; File = [arguments] : base64 BEL
		fmt.Printf("\033]1337;File=inline=1;preserveAspectRatio=1:%s\a\n", encoded)
		return true

	case kittyWindowID != "" || ghostty != "" || termProgram == "ghostty":
		// Kitty graphics protocol (also used by Ghostty)
		// Send in chunks for large images
		const chunkSize = 4096
		for i := 0; i < len(encoded); i += chunkSize {
			end := i + chunkSize
			if end > len(encoded) {
				end = len(encoded)
			}
			chunk := encoded[i:end]

			if i == 0 {
				// First chunk: a=T (transmit), f=100 (PNG), m=1 (more chunks follow) or m=0 (last chunk)
				more := 1
				if end >= len(encoded) {
					more = 0
				}
				fmt.Printf("\033_Ga=T,f=100,m=%d;%s\033\\", more, chunk)
			} else if end >= len(encoded) {
				// Last chunk
				fmt.Printf("\033_Gm=0;%s\033\\", chunk)
			} else {
				// Middle chunk
				fmt.Printf("\033_Gm=1;%s\033\\", chunk)
			}
		}
		fmt.Println()
		return true

	default:
		return false
	}
}

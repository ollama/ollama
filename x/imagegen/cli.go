// cli.go provides CLI commands for image generation models.
//
// TODO (jmorganca): Integrate these commands into cmd/cmd.go when stable.
// Currently these are separate to keep experimental code isolated.

package imagegen

import (
	"encoding/base64"
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
		Steps:  0, // 0 means model default
		Seed:   0, // 0 means random
	}
}

// RegisterFlags adds image generation flags to the given command.
// Flags are hidden since they only apply to image generation models.
func RegisterFlags(cmd *cobra.Command) {
	cmd.Flags().Int("width", 1024, "Image width")
	cmd.Flags().Int("height", 1024, "Image height")
	cmd.Flags().Int("steps", 0, "Denoising steps (0 = model default)")
	cmd.Flags().Int("seed", 0, "Random seed (0 for random)")
	cmd.Flags().String("negative", "", "Negative prompt")
	// Hide from main flags section - shown in separate section via AppendFlagsDocs
	cmd.Flags().MarkHidden("width")
	cmd.Flags().MarkHidden("height")
	cmd.Flags().MarkHidden("steps")
	cmd.Flags().MarkHidden("seed")
	cmd.Flags().MarkHidden("negative")
}

// AppendFlagsDocs appends image generation flags documentation to the command's usage template.
func AppendFlagsDocs(cmd *cobra.Command) {
	usage := `
Image Generation Flags (experimental):
      --width int      Image width
      --height int     Image height
      --steps int      Denoising steps
      --seed int       Random seed
      --negative str   Negative prompt
`
	cmd.SetUsageTemplate(cmd.UsageTemplate() + usage)
}

// RunCLI handles the CLI for image generation models.
// Returns true if it handled the request, false if the caller should continue with normal flow.
// Supports flags: --width, --height, --steps, --seed, --negative
func RunCLI(cmd *cobra.Command, name string, prompt string, interactive bool, keepAlive *api.Duration) error {
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

	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Width:  int32(opts.Width),
		Height: int32(opts.Height),
		Steps:  int32(opts.Steps),
	}
	if opts.Seed != 0 {
		req.Options = map[string]any{"seed": opts.Seed}
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}

	// Show loading spinner until generation starts
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var stepBar *progress.StepBar
	var imageBase64 string
	err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
		// Handle progress updates using structured fields
		if resp.Total > 0 {
			if stepBar == nil {
				spinner.Stop()
				stepBar = progress.NewStepBar("Generating", int(resp.Total))
				p.Add("", stepBar)
			}
			stepBar.Set(int(resp.Completed))
		}

		// Handle final response with image data
		if resp.Done && resp.Image != "" {
			imageBase64 = resp.Image
		}

		return nil
	})

	p.StopAndClear()
	if err != nil {
		return err
	}

	if imageBase64 != "" {
		// Decode base64 and save to CWD
		imageData, err := base64.StdEncoding.DecodeString(imageBase64)
		if err != nil {
			return fmt.Errorf("failed to decode image: %w", err)
		}

		// Create filename from prompt
		safeName := sanitizeFilename(prompt)
		if len(safeName) > 50 {
			safeName = safeName[:50]
		}
		timestamp := time.Now().Format("20060102-150405")
		filename := fmt.Sprintf("%s-%s.png", safeName, timestamp)

		if err := os.WriteFile(filename, imageData, 0o644); err != nil {
			return fmt.Errorf("failed to save image: %w", err)
		}

		displayImageInTerminal(filename)
		fmt.Printf("Image saved to: %s\n", filename)
	}

	return nil
}

// runInteractive runs an interactive REPL for image generation.
func runInteractive(cmd *cobra.Command, modelName string, keepAlive *api.Duration, opts ImageGenOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Preload the model with the specified keepalive
	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	preloadReq := &api.GenerateRequest{
		Model:     modelName,
		KeepAlive: keepAlive,
	}
	if err := client.Generate(cmd.Context(), preloadReq, func(resp api.GenerateResponse) error {
		return nil
	}); err != nil {
		p.StopAndClear()
		return fmt.Errorf("failed to load model: %w", err)
	}
	p.StopAndClear()

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
			printInteractiveHelp()
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
			Width:  int32(opts.Width),
			Height: int32(opts.Height),
			Steps:  int32(opts.Steps),
		}
		if opts.Seed != 0 {
			req.Options = map[string]any{"seed": opts.Seed}
		}
		if keepAlive != nil {
			req.KeepAlive = keepAlive
		}

		// Show loading spinner until generation starts
		p := progress.NewProgress(os.Stderr)
		spinner := progress.NewSpinner("")
		p.Add("", spinner)

		var stepBar *progress.StepBar
		var imageBase64 string

		err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
			// Handle progress updates using structured fields
			if resp.Total > 0 {
				if stepBar == nil {
					spinner.Stop()
					stepBar = progress.NewStepBar("Generating", int(resp.Total))
					p.Add("", stepBar)
				}
				stepBar.Set(int(resp.Completed))
			}

			// Handle final response with image data
			if resp.Done && resp.Image != "" {
				imageBase64 = resp.Image
			}

			return nil
		})

		p.StopAndClear()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		// Save image to current directory with descriptive name
		if imageBase64 != "" {
			// Decode base64 image data
			imageData, err := base64.StdEncoding.DecodeString(imageBase64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error decoding image: %v\n", err)
				continue
			}

			// Create filename from prompt (sanitized)
			safeName := sanitizeFilename(line)
			if len(safeName) > 50 {
				safeName = safeName[:50]
			}
			timestamp := time.Now().Format("20060102-150405")
			filename := fmt.Sprintf("%s-%s.png", safeName, timestamp)

			if err := os.WriteFile(filename, imageData, 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "Error saving image: %v\n", err)
				continue
			}

			displayImageInTerminal(filename)
			fmt.Printf("Image saved to: %s\n", filename)
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

// printInteractiveHelp prints help for interactive mode commands.
// TODO: reconcile /set commands with /set parameter in text gen REPL (cmd/cmd.go)
func printInteractiveHelp() {
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  /set width <n>     Set image width")
	fmt.Fprintln(os.Stderr, "  /set height <n>    Set image height")
	fmt.Fprintln(os.Stderr, "  /set steps <n>     Set denoising steps")
	fmt.Fprintln(os.Stderr, "  /set seed <n>      Set random seed")
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
			end := min(i+chunkSize, len(encoded))
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

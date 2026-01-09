package imagegen

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
)

// RunCLI handles the CLI for image generation models.
// Returns true if it handled the request, false if the caller should continue with normal flow.
func RunCLI(cmd *cobra.Command, name string, prompt string, interactive bool, keepAlive *api.Duration) error {
	// Verify it's a valid image gen model
	if ResolveModelName(name) == "" {
		return fmt.Errorf("unknown image generation model: %s", name)
	}

	if interactive {
		return runInteractive(cmd, name, keepAlive)
	}

	// One-shot generation
	return generateImage(cmd, name, prompt, keepAlive)
}

// generateImage generates a single image and displays it.
func generateImage(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
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
func runInteractive(cmd *cobra.Command, modelName string, keepAlive *api.Duration) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:      ">>> ",
		Placeholder: "Describe an image to generate (/bye to exit)",
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
			fmt.Fprintln(os.Stderr, "Type a description to generate an image.")
			fmt.Fprintln(os.Stderr, "Use /bye to exit.")
			fmt.Fprintln(os.Stderr)
			continue
		case strings.HasPrefix(line, "/"):
			fmt.Fprintf(os.Stderr, "Unknown command: %s\n", line)
			continue
		}

		// Generate image
		req := &api.GenerateRequest{
			Model:  modelName,
			Prompt: line,
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

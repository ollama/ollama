package cmd

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/types/errtypes"
)

type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
)

// MediaType defines the type of media
type MediaType int

const (
	MediaTypeImage MediaType = iota
	MediaTypeAudio
	MediaTypeVideo
)

// mediaConfig defines configuration for each media type
type mediaConfig struct {
	extensions  []string // Supported file extensions
	mimeTypes   []string // Supported MIME types
	maxSize     int64    // Maximum file size in bytes
	description string   // Media type description
}

func generateInteractive(cmd *cobra.Command, opts runOptions) error {
	usage := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set            Set session variables")
		fmt.Fprintln(os.Stderr, "  /show           Show model information")
		fmt.Fprintln(os.Stderr, "  /load <model>   Load a session or model")
		fmt.Fprintln(os.Stderr, "  /save <model>   Save your current session")
		fmt.Fprintln(os.Stderr, "  /clear          Clear session context")
		fmt.Fprintln(os.Stderr, "  /bye            Exit")
		fmt.Fprintln(os.Stderr, "  /?, /help       Help for a command")
		fmt.Fprintln(os.Stderr, "  /? shortcuts    Help for keyboard shortcuts")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Use \"\"\" to begin a multi-line message.")

		if opts.MultiModal {
			fmt.Fprintf(os.Stderr, "Use %s to include .jpg or .png images.\n", filepath.FromSlash("/path/to/file"))
		}

		fmt.Fprintln(os.Stderr, "")
	}

	usageSet := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /set parameter ...     Set a parameter")
		fmt.Fprintln(os.Stderr, "  /set system <string>   Set system message")
		fmt.Fprintln(os.Stderr, "  /set history           Enable history")
		fmt.Fprintln(os.Stderr, "  /set nohistory         Disable history")
		fmt.Fprintln(os.Stderr, "  /set wordwrap          Enable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set nowordwrap        Disable wordwrap")
		fmt.Fprintln(os.Stderr, "  /set format json       Enable JSON mode")
		fmt.Fprintln(os.Stderr, "  /set noformat          Disable formatting")
		fmt.Fprintln(os.Stderr, "  /set verbose           Show LLM stats")
		fmt.Fprintln(os.Stderr, "  /set quiet             Disable LLM stats")
		fmt.Fprintln(os.Stderr, "")
	}

	usageShortcuts := func() {
		fmt.Fprintln(os.Stderr, "Available keyboard shortcuts:")
		fmt.Fprintln(os.Stderr, "  Ctrl + a            Move to the beginning of the line (Home)")
		fmt.Fprintln(os.Stderr, "  Ctrl + e            Move to the end of the line (End)")
		fmt.Fprintln(os.Stderr, "   Alt + b            Move back (left) one word")
		fmt.Fprintln(os.Stderr, "   Alt + f            Move forward (right) one word")
		fmt.Fprintln(os.Stderr, "  Ctrl + k            Delete the sentence after the cursor")
		fmt.Fprintln(os.Stderr, "  Ctrl + u            Delete the sentence before the cursor")
		fmt.Fprintln(os.Stderr, "  Ctrl + w            Delete the word before the cursor")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  Ctrl + l            Clear the screen")
		fmt.Fprintln(os.Stderr, "  Ctrl + c            Stop the model from responding")
		fmt.Fprintln(os.Stderr, "  Ctrl + d            Exit ollama (/bye)")
		fmt.Fprintln(os.Stderr, "")
	}

	usageShow := func() {
		fmt.Fprintln(os.Stderr, "Available Commands:")
		fmt.Fprintln(os.Stderr, "  /show info         Show details for this model")
		fmt.Fprintln(os.Stderr, "  /show license      Show model license")
		fmt.Fprintln(os.Stderr, "  /show modelfile    Show Modelfile for this model")
		fmt.Fprintln(os.Stderr, "  /show parameters   Show parameters for this model")
		fmt.Fprintln(os.Stderr, "  /show system       Show system message")
		fmt.Fprintln(os.Stderr, "  /show template     Show prompt template")
		fmt.Fprintln(os.Stderr, "")
	}

	// only list out the most common parameters
	usageParameters := func() {
		fmt.Fprintln(os.Stderr, "Available Parameters:")
		fmt.Fprintln(os.Stderr, "  /set parameter seed <int>             Random number seed")
		fmt.Fprintln(os.Stderr, "  /set parameter num_predict <int>      Max number of tokens to predict")
		fmt.Fprintln(os.Stderr, "  /set parameter top_k <int>            Pick from top k num of tokens")
		fmt.Fprintln(os.Stderr, "  /set parameter top_p <float>          Pick token based on sum of probabilities")
		fmt.Fprintln(os.Stderr, "  /set parameter min_p <float>          Pick token based on top token probability * min_p")
		fmt.Fprintln(os.Stderr, "  /set parameter num_ctx <int>          Set the context size")
		fmt.Fprintln(os.Stderr, "  /set parameter temperature <float>    Set creativity level")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
		fmt.Fprintln(os.Stderr, "  /set parameter num_gpu <int>          The number of layers to send to the GPU")
		fmt.Fprintln(os.Stderr, "  /set parameter stop <string> <string> ...   Set the stop parameters")
		fmt.Fprintln(os.Stderr, "")
	}

	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}

	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	var sb strings.Builder
	var multiline MultilineState

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

			scanner.Prompt.UseAlt = false
			sb.Reset()

			continue
		case err != nil:
			return err
		}

		switch {
		case multiline != MultilineNone:
			// check if there's a multiline terminating string
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				continue
			}

			switch multiline {
			case MultilineSystem:
				opts.System = sb.String()
				opts.Messages = append(opts.Messages, api.Message{Role: "system", Content: opts.System})
				fmt.Println("Set system message.")
				sb.Reset()
			}

			multiline = MultilineNone
			scanner.Prompt.UseAlt = false
		case strings.HasPrefix(line, `"""`):
			line := strings.TrimPrefix(line, `"""`)
			line, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(line)
			if !ok {
				// no multiline terminating string; need more input
				fmt.Fprintln(&sb)
				multiline = MultilinePrompt
				scanner.Prompt.UseAlt = true
			}
		case scanner.Pasting:
			fmt.Fprintln(&sb, line)
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/load"):
			args := strings.Fields(line)
			if len(args) != 2 {
				fmt.Println("Usage:\n  /load <modelname>")
				continue
			}
			opts.Model = args[1]
			opts.Messages = []api.Message{}
			fmt.Printf("Loading model '%s'\n", opts.Model)
			if err := loadOrUnloadModel(cmd, &opts); err != nil {
				if strings.Contains(err.Error(), "not found") {
					fmt.Printf("error: %v\n", err)
					continue
				}
				return err
			}
			continue
		case strings.HasPrefix(line, "/save"):
			args := strings.Fields(line)
			if len(args) != 2 {
				fmt.Println("Usage:\n  /save <modelname>")
				continue
			}

			client, err := api.ClientFromEnvironment()
			if err != nil {
				fmt.Println("error: couldn't connect to ollama server")
				return err
			}

			req := NewCreateRequest(args[1], opts)
			fn := func(resp api.ProgressResponse) error { return nil }
			err = client.Create(cmd.Context(), req, fn)
			if err != nil {
				if strings.Contains(err.Error(), errtypes.InvalidModelNameErrMsg) {
					fmt.Printf("error: The model name '%s' is invalid\n", args[1])
					continue
				}
				return err
			}
			fmt.Printf("Created new model '%s'\n", args[1])
			continue
		case strings.HasPrefix(line, "/clear"):
			opts.Messages = []api.Message{}
			if opts.System != "" {
				newMessage := api.Message{Role: "system", Content: opts.System}
				opts.Messages = append(opts.Messages, newMessage)
			}
			fmt.Println("Cleared session context")
			continue
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "history":
					scanner.HistoryEnable()
				case "nohistory":
					scanner.HistoryDisable()
				case "wordwrap":
					opts.WordWrap = true
					fmt.Println("Set 'wordwrap' mode.")
				case "nowordwrap":
					opts.WordWrap = false
					fmt.Println("Set 'nowordwrap' mode.")
				case "verbose":
					if err := cmd.Flags().Set("verbose", "true"); err != nil {
						return err
					}
					fmt.Println("Set 'verbose' mode.")
				case "quiet":
					if err := cmd.Flags().Set("verbose", "false"); err != nil {
						return err
					}
					fmt.Println("Set 'quiet' mode.")
				case "format":
					if len(args) < 3 || args[2] != "json" {
						fmt.Println("Invalid or missing format. For 'json' mode use '/set format json'")
					} else {
						opts.Format = args[2]
						fmt.Printf("Set format to '%s' mode.\n", args[2])
					}
				case "noformat":
					opts.Format = ""
					fmt.Println("Disabled format.")
				case "parameter":
					if len(args) < 4 {
						usageParameters()
						continue
					}
					params := args[3:]
					fp, err := api.FormatParams(map[string][]string{args[2]: params})
					if err != nil {
						fmt.Printf("Couldn't set parameter: %q\n", err)
						continue
					}
					fmt.Printf("Set parameter '%s' to '%s'\n", args[2], strings.Join(params, ", "))
					opts.Options[args[2]] = fp[args[2]]
				case "system":
					if len(args) < 3 {
						usageSet()
						continue
					}

					multiline = MultilineSystem

					line := strings.Join(args[2:], " ")
					line, ok := strings.CutPrefix(line, `"""`)
					if !ok {
						multiline = MultilineNone
					} else {
						// only cut suffix if the line is multiline
						line, ok = strings.CutSuffix(line, `"""`)
						if ok {
							multiline = MultilineNone
						}
					}

					sb.WriteString(line)
					if multiline != MultilineNone {
						scanner.Prompt.UseAlt = true
						continue
					}

					opts.System = sb.String() // for display in modelfile
					newMessage := api.Message{Role: "system", Content: sb.String()}
					// Check if the slice is not empty and the last message is from 'system'
					if len(opts.Messages) > 0 && opts.Messages[len(opts.Messages)-1].Role == "system" {
						// Replace the last message
						opts.Messages[len(opts.Messages)-1] = newMessage
					} else {
						opts.Messages = append(opts.Messages, newMessage)
					}
					fmt.Println("Set system message.")
					sb.Reset()
					continue
				default:
					fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
				}
			} else {
				usageSet()
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) > 1 {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					fmt.Println("error: couldn't connect to ollama server")
					return err
				}
				req := &api.ShowRequest{
					Name:    opts.Model,
					System:  opts.System,
					Options: opts.Options,
				}
				resp, err := client.Show(cmd.Context(), req)
				if err != nil {
					fmt.Println("error: couldn't get model")
					return err
				}

				switch args[1] {
				case "info":
					_ = showInfo(resp, os.Stderr)
				case "license":
					if resp.License == "" {
						fmt.Println("No license was specified for this model.")
					} else {
						fmt.Println(resp.License)
					}
				case "modelfile":
					fmt.Println(resp.Modelfile)
				case "parameters":
					if resp.Parameters == "" {
						fmt.Println("No parameters were specified for this model.")
					} else {
						if len(opts.Options) > 0 {
							fmt.Println("User defined parameters:")
							for k, v := range opts.Options {
								fmt.Printf("%-*s %v\n", 30, k, v)
							}
							fmt.Println()
						}
						fmt.Println("Model defined parameters:")
						fmt.Println(resp.Parameters)
					}
				case "system":
					switch {
					case opts.System != "":
						fmt.Println(opts.System + "\n")
					case resp.System != "":
						fmt.Println(resp.System + "\n")
					default:
						fmt.Println("No system message was specified for this model.")
					}
				case "template":
					if resp.Template != "" {
						fmt.Println(resp.Template)
					} else {
						fmt.Println("No prompt template was specified for this model.")
					}
				default:
					fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
				}
			} else {
				usageShow()
			}
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "set", "/set":
					usageSet()
				case "show", "/show":
					usageShow()
				case "shortcut", "shortcuts":
					usageShortcuts()
				}
			} else {
				usage()
			}
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			isFile := false

			if opts.MultiModal {
				for _, f := range extractFileNames(line, MediaTypeImage) {
					if strings.HasPrefix(f, args[0]) {
						isFile = true
						break
					}
				}
			}

			if !isFile {
				fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
				continue
			}

			sb.WriteString(line)
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == MultilineNone {
			newMessage := api.Message{Role: "user", Content: sb.String()}

			if opts.MultiModal {
				msg, images, imageUrls, audioUrls, videoUrls, err := extractFileData(sb.String())
				if err != nil {
					return err
				}

				newMessage.Content = msg
				newMessage.Images = images
				newMessage.ImageUrls = imageUrls
				newMessage.AudioUrls = audioUrls
				newMessage.VideoUrls = videoUrls
			}

			opts.Messages = append(opts.Messages, newMessage)

			assistant, err := chat(cmd, opts)
			if err != nil {
				return err
			}
			if assistant != nil {
				opts.Messages = append(opts.Messages, *assistant)
			}

			sb.Reset()
		}
	}
}

func NewCreateRequest(name string, opts runOptions) *api.CreateRequest {
	req := &api.CreateRequest{
		Name: name,
		From: cmp.Or(opts.ParentModel, opts.Model),
	}

	if opts.System != "" {
		req.System = opts.System
	}

	if len(opts.Options) > 0 {
		req.Parameters = opts.Options
	}

	if len(opts.Messages) > 0 {
		req.Messages = opts.Messages
	}

	return req
}

func normalizeFilePath(fp string) string {
	return strings.NewReplacer(
		"\\ ", " ", // Escaped space
		"\\(", "(", // Escaped left parenthesis
		"\\)", ")", // Escaped right parenthesis
		"\\[", "[", // Escaped left square bracket
		"\\]", "]", // Escaped right square bracket
		"\\{", "{", // Escaped left curly brace
		"\\}", "}", // Escaped right curly brace
		"\\$", "$", // Escaped dollar sign
		"\\&", "&", // Escaped ampersand
		"\\;", ";", // Escaped semicolon
		"\\'", "'", // Escaped single quote
		"\\\\", "\\", // Escaped backslash
		"\\*", "*", // Escaped asterisk
		"\\?", "?", // Escaped question mark
	).Replace(fp)
}

func extractFileNames(input string, mediaType MediaType) []string {
	config := mediaConfigs[mediaType]
	// Convert extension array to regex pattern
	extensions := strings.Join(config.extensions, "|")
	// Build regex pattern: match paths starting with optional drive letter,
	// followed by / ./ \ or .\ and ending with specified extensions (case insensitive)
	regexPattern := fmt.Sprintf(`(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:%s)\b`, extensions)
	re := regexp.MustCompile(regexPattern)

	// Find all matching file paths
	matches := re.FindAllString(input, -1)
	if matches == nil {
		return []string{}
	}
	return matches
}

func extractFileData(input string) (string, []api.ImageData, []string, []string, []string, error) {
	var images []api.ImageData
	var imageUrls []string
	var audioUrls []string
	var videoUrls []string

	for mediaType := range mediaConfigs {
		paths := extractFileNames(input, mediaType)
		for _, fp := range paths {
			nfp := normalizeFilePath(fp)
			if _, err := os.Stat(nfp); errors.Is(err, os.ErrNotExist) {
				fmt.Fprintf(os.Stderr, "File '%s' does not exist\n", nfp)
				continue
			}

			switch mediaType {
			case MediaTypeImage:
				data, err := getMediaData(nfp, mediaType)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Read image '%s' failed. Error is %s.\n", nfp, err.Error())
				}
				images = append(images, api.ImageData(data))
				imageUrls = append(imageUrls, nfp)
				fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
			case MediaTypeAudio:
				audioUrls = append(audioUrls, nfp)
				fmt.Fprintf(os.Stderr, "Added audio '%s'\n", nfp)
			case MediaTypeVideo:
				videoUrls = append(videoUrls, nfp)
				fmt.Fprintf(os.Stderr, "Added video '%s'\n", nfp)
			}

			input = strings.ReplaceAll(input, fp, "")
		}
	}

	return strings.TrimSpace(input), images, imageUrls, audioUrls, videoUrls, nil
}

// getMediaData reads and validates media files
func getMediaData(filePath string, mediaType MediaType) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read file header to detect type
	header := make([]byte, 512)
	_, err = file.Read(header)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("failed to read file header: %w", err)
	}

	// Get file info
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	// Validate file type and size
	contentType := http.DetectContentType(header)
	config := mediaConfigs[mediaType]

	// Validate file type
	validType := false
	for _, mimeType := range config.mimeTypes {
		if strings.HasPrefix(contentType, mimeType) {
			validType = true
			break
		}
	}
	if !validType {
		return nil, fmt.Errorf("unsupported %s type: %s", config.description, contentType)
	}

	// Validate file size
	if info.Size() > config.maxSize {
		return nil, fmt.Errorf("%s file size exceeds limit (max %dMB)",
			config.description, config.maxSize/1024/1024)
	}

	// Read entire file content
	buf := make([]byte, info.Size())
	_, err = file.Seek(0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to reset file pointer: %w", err)
	}

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, fmt.Errorf("failed to read file content: %w", err)
	}

	return buf, nil
}

// mediaConfigs stores configurations for all media types
var mediaConfigs = map[MediaType]mediaConfig{
	MediaTypeImage: {
		extensions:  []string{"jpg", "jpeg", "png"},
		mimeTypes:   []string{"image/jpeg", "image/png"},
		maxSize:     100 * 1024 * 1024, // 100MB
		description: "image",
	},
	MediaTypeAudio: {
		extensions: []string{"mp3"},
		mimeTypes: []string{
			"audio/mpeg",
			"audio/wav",
			"audio/ogg",
			"audio/mp4",
			"audio/x-m4a",
			"audio/aac",
			"audio/webm",
			"application/ogg",
		},
		maxSize:     50 * 1024 * 1024, // 50MB
		description: "audio",
	},
	MediaTypeVideo: {
		extensions: []string{"mp4"},
		mimeTypes: []string{
			"video/mp4",
			"video/x-msvideo",
			"video/quicktime",
			"video/x-matroska",
			"video/webm",
			"video/mpeg",
			"video/ogg",
			"application/x-mpegURL",
			"video/MP2T",
		},
		maxSize:     200 * 1024 * 1024, // 200MB
		description: "video",
	},
}

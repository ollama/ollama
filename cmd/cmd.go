package cmd

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/containerd/console"
	"github.com/mattn/go-runewidth"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"golang.org/x/sync/errgroup"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/runner"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/types/syncmap"
	"github.com/ollama/ollama/version"
	xcmd "github.com/ollama/ollama/x/cmd"
	"github.com/ollama/ollama/x/create"
	xcreateclient "github.com/ollama/ollama/x/create/client"
	"github.com/ollama/ollama/x/imagegen"
)

const ConnectInstructions = "To sign in, navigate to:\n    %s\n\n"

// ensureThinkingSupport emits a warning if the model does not advertise thinking support
func ensureThinkingSupport(ctx context.Context, client *api.Client, name string) {
	if name == "" {
		return
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		return
	}
	if slices.Contains(resp.Capabilities, model.CapabilityThinking) {
		return
	}
	fmt.Fprintf(os.Stderr, "warning: model %q does not support thinking output\n", name)
}

var errModelfileNotFound = errors.New("specified Modelfile wasn't found")

func getModelfileName(cmd *cobra.Command) (string, error) {
	filename, _ := cmd.Flags().GetString("file")

	if filename == "" {
		filename = "Modelfile"
	}

	absName, err := filepath.Abs(filename)
	if err != nil {
		return "", err
	}

	_, err = os.Stat(absName)
	if err != nil {
		return "", err
	}

	return absName, nil
}

func CreateHandler(cmd *cobra.Command, args []string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	// Validate model name early to fail fast
	modelName := args[0]
	name := model.ParseName(modelName)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	// Check for --experimental flag for safetensors model creation
	experimental, _ := cmd.Flags().GetBool("experimental")
	if experimental {
		// Get Modelfile content - either from -f flag or default to "FROM ."
		var reader io.Reader
		filename, err := getModelfileName(cmd)
		if os.IsNotExist(err) || filename == "" {
			// No Modelfile specified or found - use default
			reader = strings.NewReader("FROM .\n")
		} else if err != nil {
			return err
		} else {
			f, err := os.Open(filename)
			if err != nil {
				return err
			}
			defer f.Close()
			reader = f
		}

		// Parse the Modelfile
		modelfile, err := parser.ParseFile(reader)
		if err != nil {
			return fmt.Errorf("failed to parse Modelfile: %w", err)
		}

		// Extract FROM path and configuration
		var modelDir string
		mfConfig := &xcreateclient.ModelfileConfig{}

		for _, cmd := range modelfile.Commands {
			switch cmd.Name {
			case "model":
				modelDir = cmd.Args
			case "template":
				mfConfig.Template = cmd.Args
			case "system":
				mfConfig.System = cmd.Args
			case "license":
				mfConfig.License = cmd.Args
			}
		}

		if modelDir == "" {
			modelDir = "."
		}

		// Resolve relative paths based on Modelfile location
		if !filepath.IsAbs(modelDir) && filename != "" {
			modelDir = filepath.Join(filepath.Dir(filename), modelDir)
		}

		quantize, _ := cmd.Flags().GetString("quantize")
		return xcreateclient.CreateModel(xcreateclient.CreateOptions{
			ModelName: modelName,
			ModelDir:  modelDir,
			Quantize:  quantize,
			Modelfile: mfConfig,
		}, p)
	}

	var reader io.Reader

	filename, err := getModelfileName(cmd)
	if os.IsNotExist(err) {
		if filename == "" {
			// No Modelfile found - check if current directory is an image gen model
			if create.IsTensorModelDir(".") {
				quantize, _ := cmd.Flags().GetString("quantize")
				return xcreateclient.CreateModel(xcreateclient.CreateOptions{
					ModelName: modelName,
					ModelDir:  ".",
					Quantize:  quantize,
				}, p)
			}
			reader = strings.NewReader("FROM .\n")
		} else {
			return errModelfileNotFound
		}
	} else if err != nil {
		return err
	} else {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}

		reader = f
		defer f.Close()
	}

	modelfile, err := parser.ParseFile(reader)
	if err != nil {
		return err
	}

	status := "gathering model components"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	req, err := modelfile.CreateRequest(filepath.Dir(filename))
	if err != nil {
		return err
	}
	spinner.Stop()

	req.Model = modelName
	quantize, _ := cmd.Flags().GetString("quantize")
	if quantize != "" {
		req.Quantize = quantize
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	var g errgroup.Group
	g.SetLimit(max(runtime.GOMAXPROCS(0)-1, 1))

	files := syncmap.NewSyncMap[string, string]()
	for f, digest := range req.Files {
		g.Go(func() error {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}

			// TODO: this is incorrect since the file might be in a subdirectory
			//       instead this should take the path relative to the model directory
			//       but the current implementation does not allow this
			files.Store(filepath.Base(f), digest)
			return nil
		})
	}

	adapters := syncmap.NewSyncMap[string, string]()
	for f, digest := range req.Adapters {
		g.Go(func() error {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}

			// TODO: same here
			adapters.Store(filepath.Base(f), digest)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	req.Files = files.Items()
	req.Adapters = adapters.Items()

	bars := make(map[string]*progress.Bar)
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("pulling %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			spinner.Stop()

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	if err := client.Create(cmd.Context(), req, fn); err != nil {
		if strings.Contains(err.Error(), "path or Modelfile are required") {
			return fmt.Errorf("the ollama server must be updated to use `ollama create` with this client")
		}
		return err
	}

	return nil
}

func createBlob(cmd *cobra.Command, client *api.Client, path string, digest string, p *progress.Progress) (string, error) {
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}

	bin, err := os.Open(realPath)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	// Get file info to retrieve the size
	fileInfo, err := bin.Stat()
	if err != nil {
		return "", err
	}
	fileSize := fileInfo.Size()

	var pw progressWriter
	status := fmt.Sprintf("copying file %s 0%%", digest)
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)
	defer spinner.Stop()

	done := make(chan struct{})
	defer close(done)

	go func() {
		ticker := time.NewTicker(60 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				spinner.SetMessage(fmt.Sprintf("copying file %s %d%%", digest, int(100*pw.n.Load()/fileSize)))
			case <-done:
				spinner.SetMessage(fmt.Sprintf("copying file %s 100%%", digest))
				return
			}
		}
	}()

	if err := client.CreateBlob(cmd.Context(), digest, io.TeeReader(bin, &pw)); err != nil {
		return "", err
	}
	return digest, nil
}

type progressWriter struct {
	n atomic.Int64
}

func (w *progressWriter) Write(p []byte) (n int, err error) {
	w.n.Add(int64(len(p)))
	return len(p), nil
}

func loadOrUnloadModel(cmd *cobra.Command, opts *runOptions) error {
	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if info, err := client.Show(cmd.Context(), &api.ShowRequest{Model: opts.Model}); err != nil {
		return err
	} else if info.RemoteHost != "" {
		// Cloud model, no need to load/unload
		if opts.ShowConnect {
			p.StopAndClear()
			if strings.HasPrefix(info.RemoteHost, "https://ollama.com") {
				fmt.Fprintf(os.Stderr, "Connecting to '%s' on 'ollama.com' ⚡\n", info.RemoteModel)
			} else {
				fmt.Fprintf(os.Stderr, "Connecting to '%s' on '%s'\n", info.RemoteModel, info.RemoteHost)
			}
		}
		return nil
	}

	req := &api.GenerateRequest{
		Model:     opts.Model,
		KeepAlive: opts.KeepAlive,

		// pass Think here so we fail before getting to the chat prompt if the model doesn't support it
		Think: opts.Think,
	}

	return client.Generate(cmd.Context(), req, func(r api.GenerateResponse) error {
		return nil
	})
}

func StopHandler(cmd *cobra.Command, args []string) error {
	opts := &runOptions{
		Model:     args[0],
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := loadOrUnloadModel(cmd, opts); err != nil {
		if strings.Contains(err.Error(), "not found") {
			return fmt.Errorf("couldn't find model \"%s\" to stop", args[0])
		}
		return err
	}
	return nil
}

func generateEmbedding(cmd *cobra.Command, modelName, input string, keepAlive *api.Duration, truncate *bool, dimensions int) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := &api.EmbedRequest{
		Model: modelName,
		Input: input,
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}
	if truncate != nil {
		req.Truncate = truncate
	}
	if dimensions > 0 {
		req.Dimensions = dimensions
	}

	resp, err := client.Embed(cmd.Context(), req)
	if err != nil {
		return err
	}

	if len(resp.Embeddings) == 0 {
		return errors.New("no embeddings returned")
	}

	output, err := json.Marshal(resp.Embeddings[0])
	if err != nil {
		return err
	}
	fmt.Println(string(output))

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	interactive := true

	opts := runOptions{
		Model:       args[0],
		WordWrap:    os.Getenv("TERM") == "xterm-256color",
		Options:     map[string]any{},
		ShowConnect: true,
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	thinkFlag := cmd.Flags().Lookup("think")
	if thinkFlag.Changed {
		thinkStr, err := cmd.Flags().GetString("think")
		if err != nil {
			return err
		}

		// Handle different values for --think
		switch thinkStr {
		case "", "true":
			// --think or --think=true
			opts.Think = &api.ThinkValue{Value: true}
		case "false":
			opts.Think = &api.ThinkValue{Value: false}
		case "high", "medium", "low":
			opts.Think = &api.ThinkValue{Value: thinkStr}
		default:
			return fmt.Errorf("invalid value for --think: %q (must be true, false, high, medium, or low)", thinkStr)
		}
	} else {
		opts.Think = nil
	}
	hidethinking, err := cmd.Flags().GetBool("hidethinking")
	if err != nil {
		return err
	}
	opts.HideThinking = hidethinking

	keepAlive, err := cmd.Flags().GetString("keepalive")
	if err != nil {
		return err
	}
	if keepAlive != "" {
		d, err := time.ParseDuration(keepAlive)
		if err != nil {
			return err
		}
		opts.KeepAlive = &api.Duration{Duration: d}
	}

	prompts := args[1:]
	// prepend stdin to the prompt if provided
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		in, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}

		// Only prepend stdin content if it's not empty
		stdinContent := string(in)
		if len(stdinContent) > 0 {
			prompts = append([]string{stdinContent}, prompts...)
		}
		opts.ShowConnect = false
		opts.WordWrap = false
		interactive = false
	}
	opts.Prompt = strings.Join(prompts, " ")
	if len(prompts) > 0 {
		interactive = false
	}
	// Be quiet if we're redirecting to a pipe or file
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		interactive = false
	}

	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	opts.WordWrap = !nowrap

	// Fill out the rest of the options based on information about the
	// model.
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	name := args[0]

	info, err := func() (*api.ShowResponse, error) {
		showReq := &api.ShowRequest{Name: name}
		info, err := client.Show(cmd.Context(), showReq)
		var se api.StatusError
		if errors.As(err, &se) && se.StatusCode == http.StatusNotFound {
			if err := PullHandler(cmd, []string{name}); err != nil {
				return nil, err
			}
			return client.Show(cmd.Context(), &api.ShowRequest{Name: name})
		}
		return info, err
	}()
	if err != nil {
		return err
	}

	opts.Think, err = inferThinkingOption(&info.Capabilities, &opts, thinkFlag.Changed)
	if err != nil {
		return err
	}

	opts.MultiModal = slices.Contains(info.Capabilities, model.CapabilityVision)

	// TODO: remove the projector info and vision info checks below,
	// these are left in for backwards compatibility with older servers
	// that don't have the capabilities field in the model info
	if len(info.ProjectorInfo) != 0 {
		opts.MultiModal = true
	}
	for k := range info.ModelInfo {
		if strings.Contains(k, ".vision.") {
			opts.MultiModal = true
			break
		}
	}

	opts.ParentModel = info.Details.ParentModel

	// Check if this is an embedding model
	isEmbeddingModel := slices.Contains(info.Capabilities, model.CapabilityEmbedding)

	// If it's an embedding model, handle embedding generation
	if isEmbeddingModel {
		if opts.Prompt == "" {
			return errors.New("embedding models require input text. Usage: ollama run " + name + " \"your text here\"")
		}

		// Get embedding-specific flags
		var truncate *bool
		if truncateFlag, err := cmd.Flags().GetBool("truncate"); err == nil && cmd.Flags().Changed("truncate") {
			truncate = &truncateFlag
		}

		dimensions, err := cmd.Flags().GetInt("dimensions")
		if err != nil {
			return err
		}

		return generateEmbedding(cmd, name, opts.Prompt, opts.KeepAlive, truncate, dimensions)
	}

	// Check if this is an image generation model
	if slices.Contains(info.Capabilities, model.CapabilityImage) {
		if opts.Prompt == "" && !interactive {
			return errors.New("image generation models require a prompt. Usage: ollama run " + name + " \"your prompt here\"")
		}
		return imagegen.RunCLI(cmd, name, opts.Prompt, interactive, opts.KeepAlive)
	}

	// Check for experimental flag
	isExperimental, _ := cmd.Flags().GetBool("experimental")
	yoloMode, _ := cmd.Flags().GetBool("experimental-yolo")
	enableWebsearch, _ := cmd.Flags().GetBool("experimental-websearch")

	if interactive {
		if err := loadOrUnloadModel(cmd, &opts); err != nil {
			var sErr api.AuthorizationError
			if errors.As(err, &sErr) && sErr.StatusCode == http.StatusUnauthorized {
				fmt.Printf("You need to be signed in to Ollama to run Cloud models.\n\n")

				if sErr.SigninURL != "" {
					fmt.Printf(ConnectInstructions, sErr.SigninURL)
				}
				return nil
			}
			return err
		}

		for _, msg := range info.Messages {
			switch msg.Role {
			case "user":
				fmt.Printf(">>> %s\n", msg.Content)
			case "assistant":
				state := &displayResponseState{}
				displayResponse(msg.Content, opts.WordWrap, state)
				fmt.Println()
				fmt.Println()
			}
		}

		// Use experimental agent loop with tools
		if isExperimental {
			return xcmd.GenerateInteractive(cmd, opts.Model, opts.WordWrap, opts.Options, opts.Think, opts.HideThinking, opts.KeepAlive, yoloMode, enableWebsearch)
		}

		return generateInteractive(cmd, opts)
	}
	return generate(cmd, opts)
}

func SigninHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	user, err := client.Whoami(cmd.Context())
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
			fmt.Println("You need to be signed in to Ollama to run Cloud models.")
			fmt.Println()

			if aErr.SigninURL != "" {
				fmt.Printf(ConnectInstructions, aErr.SigninURL)
			}
			return nil
		}
		return err
	}

	if user != nil && user.Name != "" {
		fmt.Printf("You are already signed in as user '%s'\n", user.Name)
		fmt.Println()
		return nil
	}

	return nil
}

func SignoutHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	err = client.Signout(cmd.Context())
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
			fmt.Println("You are not signed in to ollama.com")
			fmt.Println()
			return nil
		} else {
			return err
		}
	}

	fmt.Println("You have signed out of ollama.com")
	fmt.Println()
	return nil
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	n := model.ParseName(args[0])
	if strings.HasSuffix(n.Host, ".ollama.ai") || strings.HasSuffix(n.Host, ".ollama.com") {
		_, err := client.Whoami(cmd.Context())
		if err != nil {
			var aErr api.AuthorizationError
			if errors.As(err, &aErr) && aErr.StatusCode == http.StatusUnauthorized {
				fmt.Println("You need to be signed in to push models to ollama.com.")
				fmt.Println()

				if aErr.SigninURL != "" {
					fmt.Printf(ConnectInstructions, aErr.SigninURL)
				}
				return nil
			}

			return err
		}
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				msg := resp.Status
				if msg == "" {
					msg = fmt.Sprintf("pushing %s...", resp.Digest[7:19])
				}
				bar = progress.NewBar(msg, resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PushRequest{Name: args[0], Insecure: insecure}

	if err := client.Push(cmd.Context(), &request, fn); err != nil {
		if spinner != nil {
			spinner.Stop()
		}
		errStr := strings.ToLower(err.Error())
		if strings.Contains(errStr, "access denied") || strings.Contains(errStr, "unauthorized") {
			return errors.New("you are not authorized to push to this namespace, create the model under a namespace you own")
		}
		return err
	}

	p.Stop()
	spinner.Stop()

	destination := n.String()
	if strings.HasSuffix(n.Host, ".ollama.ai") || strings.HasSuffix(n.Host, ".ollama.com") {
		destination = "https://ollama.com/" + strings.TrimSuffix(n.DisplayShortest(), ":latest")
	}
	fmt.Printf("\nYou can find your model at:\n\n")
	fmt.Printf("\t%s\n", destination)

	return nil
}

func ListHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	models, err := client.List(cmd.Context())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(strings.ToLower(m.Name), strings.ToLower(args[0])) {
			var size string
			if m.RemoteModel != "" {
				size = "-"
			} else {
				size = format.HumanBytes(m.Size)
			}

			data = append(data, []string{m.Name, m.Digest[:12], size, format.HumanTime(m.ModifiedAt, "Never")})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")
	table.AppendBulk(data)
	table.Render()

	return nil
}

func ListRunningHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	models, err := client.ListRunning(cmd.Context())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			var procStr string
			switch {
			case m.SizeVRAM == 0:
				procStr = "100% CPU"
			case m.SizeVRAM == m.Size:
				procStr = "100% GPU"
			case m.SizeVRAM > m.Size || m.Size == 0:
				procStr = "Unknown"
			default:
				sizeCPU := m.Size - m.SizeVRAM
				cpuPercent := math.Round(float64(sizeCPU) / float64(m.Size) * 100)
				procStr = fmt.Sprintf("%d%%/%d%% CPU/GPU", int(cpuPercent), int(100-cpuPercent))
			}

			var until string
			delta := time.Since(m.ExpiresAt)
			if delta > 0 {
				until = "Stopping..."
			} else {
				until = format.HumanTime(m.ExpiresAt, "Never")
			}
			ctxStr := strconv.Itoa(m.ContextLength)
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), procStr, ctxStr, until})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("    ")
	table.AppendBulk(data)
	table.Render()

	return nil
}

func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	for _, arg := range args {
		// Unload the model if it's running before deletion
		if err := loadOrUnloadModel(cmd, &runOptions{
			Model:     args[0],
			KeepAlive: &api.Duration{Duration: 0},
		}); err != nil {
			if !strings.Contains(strings.ToLower(err.Error()), "not found") {
				fmt.Fprintf(os.Stderr, "Warning: unable to stop model '%s'\n", args[0])
			}
		}

		if err := client.Delete(cmd.Context(), &api.DeleteRequest{Name: arg}); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", arg)
	}
	return nil
}

func ShowHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	license, errLicense := cmd.Flags().GetBool("license")
	modelfile, errModelfile := cmd.Flags().GetBool("modelfile")
	parameters, errParams := cmd.Flags().GetBool("parameters")
	system, errSystem := cmd.Flags().GetBool("system")
	template, errTemplate := cmd.Flags().GetBool("template")
	verbose, errVerbose := cmd.Flags().GetBool("verbose")

	for _, boolErr := range []error{errLicense, errModelfile, errParams, errSystem, errTemplate, errVerbose} {
		if boolErr != nil {
			return errors.New("error retrieving flags")
		}
	}

	flagsSet := 0
	showType := ""

	if license {
		flagsSet++
		showType = "license"
	}

	if modelfile {
		flagsSet++
		showType = "modelfile"
	}

	if parameters {
		flagsSet++
		showType = "parameters"
	}

	if system {
		flagsSet++
		showType = "system"
	}

	if template {
		flagsSet++
		showType = "template"
	}

	if flagsSet > 1 {
		return errors.New("only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
	}

	req := api.ShowRequest{Name: args[0], Verbose: verbose}
	resp, err := client.Show(cmd.Context(), &req)
	if err != nil {
		return err
	}

	if flagsSet == 1 {
		switch showType {
		case "license":
			fmt.Println(resp.License)
		case "modelfile":
			fmt.Println(resp.Modelfile)
		case "parameters":
			fmt.Println(resp.Parameters)
		case "system":
			fmt.Print(resp.System)
		case "template":
			fmt.Print(resp.Template)
		}

		return nil
	}

	return showInfo(resp, verbose, os.Stdout)
}

func showInfo(resp *api.ShowResponse, verbose bool, w io.Writer) error {
	tableRender := func(header string, rows func() [][]string) {
		fmt.Fprintln(w, " ", header)
		table := tablewriter.NewWriter(w)
		table.SetAlignment(tablewriter.ALIGN_LEFT)
		table.SetBorder(false)
		table.SetNoWhiteSpace(true)
		table.SetTablePadding("    ")

		switch header {
		case "Template", "System", "License":
			table.SetColWidth(100)
		}

		table.AppendBulk(rows())
		table.Render()
		fmt.Fprintln(w)
	}

	tableRender("Model", func() (rows [][]string) {
		if resp.RemoteHost != "" {
			rows = append(rows, []string{"", "Remote model", resp.RemoteModel})
			rows = append(rows, []string{"", "Remote URL", resp.RemoteHost})
		}

		if resp.ModelInfo != nil {
			arch := resp.ModelInfo["general.architecture"].(string)
			rows = append(rows, []string{"", "architecture", arch})

			var paramStr string
			if resp.Details.ParameterSize != "" {
				paramStr = resp.Details.ParameterSize
			} else if v, ok := resp.ModelInfo["general.parameter_count"]; ok {
				if f, ok := v.(float64); ok {
					paramStr = format.HumanNumber(uint64(f))
				}
			}
			rows = append(rows, []string{"", "parameters", paramStr})

			if v, ok := resp.ModelInfo[fmt.Sprintf("%s.context_length", arch)]; ok {
				if f, ok := v.(float64); ok {
					rows = append(rows, []string{"", "context length", strconv.FormatFloat(f, 'f', -1, 64)})
				}
			}

			if v, ok := resp.ModelInfo[fmt.Sprintf("%s.embedding_length", arch)]; ok {
				if f, ok := v.(float64); ok {
					rows = append(rows, []string{"", "embedding length", strconv.FormatFloat(f, 'f', -1, 64)})
				}
			}
		} else {
			rows = append(rows, []string{"", "architecture", resp.Details.Family})
			rows = append(rows, []string{"", "parameters", resp.Details.ParameterSize})
		}
		rows = append(rows, []string{"", "quantization", resp.Details.QuantizationLevel})
		if resp.Requires != "" {
			rows = append(rows, []string{"", "requires", resp.Requires})
		}
		return
	})

	if len(resp.Capabilities) > 0 {
		tableRender("Capabilities", func() (rows [][]string) {
			for _, capability := range resp.Capabilities {
				rows = append(rows, []string{"", capability.String()})
			}
			return
		})
	}

	if resp.ProjectorInfo != nil {
		tableRender("Projector", func() (rows [][]string) {
			arch := resp.ProjectorInfo["general.architecture"].(string)
			rows = append(rows, []string{"", "architecture", arch})
			rows = append(rows, []string{"", "parameters", format.HumanNumber(uint64(resp.ProjectorInfo["general.parameter_count"].(float64)))})
			rows = append(rows, []string{"", "embedding length", strconv.FormatFloat(resp.ProjectorInfo[fmt.Sprintf("%s.vision.embedding_length", arch)].(float64), 'f', -1, 64)})
			rows = append(rows, []string{"", "dimensions", strconv.FormatFloat(resp.ProjectorInfo[fmt.Sprintf("%s.vision.projection_dim", arch)].(float64), 'f', -1, 64)})
			return
		})
	}

	if resp.Parameters != "" {
		tableRender("Parameters", func() (rows [][]string) {
			scanner := bufio.NewScanner(strings.NewReader(resp.Parameters))
			for scanner.Scan() {
				if text := scanner.Text(); text != "" {
					rows = append(rows, append([]string{""}, strings.Fields(text)...))
				}
			}
			return
		})
	}

	if resp.ModelInfo != nil && verbose {
		tableRender("Metadata", func() (rows [][]string) {
			keys := make([]string, 0, len(resp.ModelInfo))
			for k := range resp.ModelInfo {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			for _, k := range keys {
				var v string
				switch vData := resp.ModelInfo[k].(type) {
				case bool:
					v = fmt.Sprintf("%t", vData)
				case string:
					v = vData
				case float64:
					v = fmt.Sprintf("%g", vData)
				case []any:
					targetWidth := 10 // Small width where we are displaying the data in a column

					var itemsToShow int
					totalWidth := 1 // Start with 1 for opening bracket

					// Find how many we can fit
					for i := range vData {
						itemStr := fmt.Sprintf("%v", vData[i])
						width := runewidth.StringWidth(itemStr)

						// Add separator width (", ") for all items except the first
						if i > 0 {
							width += 2
						}

						// Check if adding this item would exceed our width limit
						if totalWidth+width > targetWidth && i > 0 {
							break
						}

						totalWidth += width
						itemsToShow++
					}

					// Format the output
					if itemsToShow < len(vData) {
						v = fmt.Sprintf("%v", vData[:itemsToShow])
						v = strings.TrimSuffix(v, "]")
						v += fmt.Sprintf(" ...+%d more]", len(vData)-itemsToShow)
					} else {
						v = fmt.Sprintf("%v", vData)
					}
				default:
					v = fmt.Sprintf("%T", vData)
				}
				rows = append(rows, []string{"", k, v})
			}
			return
		})
	}

	if len(resp.Tensors) > 0 && verbose {
		tableRender("Tensors", func() (rows [][]string) {
			for _, t := range resp.Tensors {
				rows = append(rows, []string{"", t.Name, t.Type, fmt.Sprint(t.Shape)})
			}
			return
		})
	}

	head := func(s string, n int) (rows [][]string) {
		scanner := bufio.NewScanner(strings.NewReader(s))
		count := 0
		for scanner.Scan() {
			text := strings.TrimSpace(scanner.Text())
			if text == "" {
				continue
			}
			count++
			if n < 0 || count <= n {
				rows = append(rows, []string{"", text})
			}
		}
		if n >= 0 && count > n {
			rows = append(rows, []string{"", "..."})
		}
		return
	}

	if resp.System != "" {
		tableRender("System", func() [][]string {
			return head(resp.System, 2)
		})
	}

	if resp.License != "" {
		tableRender("License", func() [][]string {
			return head(resp.License, 2)
		})
	}

	return nil
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(cmd.Context(), &req); err != nil {
		return err
	}
	fmt.Printf("copied '%s' to '%s'\n", args[0], args[1])
	return nil
}

func PullHandler(cmd *cobra.Command, args []string) error {
	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)

	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if resp.Completed == 0 {
				// This is the initial status update for the
				// layer, which the server sends before
				// beginning the download, for clients to
				// compute total size and prepare for
				// downloads, if needed.
				//
				// Skipping this here to avoid showing a 0%
				// progress bar, which *should* clue the user
				// into the fact that many things are being
				// downloaded and that the current active
				// download is not that last. However, in rare
				// cases it seems to be triggering to some, and
				// it isn't worth explaining, so just ignore
				// and regress to the old UI that keeps giving
				// you the "But wait, there is more!" after
				// each "100% done" bar, which is "better."
				return nil
			}

			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				name, isDigest := strings.CutPrefix(resp.Digest, "sha256:")
				name = strings.TrimSpace(name)
				if isDigest {
					name = name[:min(12, len(name))]
				}
				bar = progress.NewBar(fmt.Sprintf("pulling %s:", name), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: args[0], Insecure: insecure}
	return client.Pull(cmd.Context(), &request, fn)
}

type generateContextKey string

type runOptions struct {
	Model        string
	ParentModel  string
	Prompt       string
	Messages     []api.Message
	WordWrap     bool
	Format       string
	System       string
	Images       []api.ImageData
	Options      map[string]any
	MultiModal   bool
	KeepAlive    *api.Duration
	Think        *api.ThinkValue
	HideThinking bool
	ShowConnect  bool
}

func (r runOptions) Copy() runOptions {
	var messages []api.Message
	if r.Messages != nil {
		messages = make([]api.Message, len(r.Messages))
		copy(messages, r.Messages)
	}

	var images []api.ImageData
	if r.Images != nil {
		images = make([]api.ImageData, len(r.Images))
		copy(images, r.Images)
	}

	var opts map[string]any
	if r.Options != nil {
		opts = make(map[string]any, len(r.Options))
		for k, v := range r.Options {
			opts[k] = v
		}
	}

	var think *api.ThinkValue
	if r.Think != nil {
		cThink := *r.Think
		think = &cThink
	}

	return runOptions{
		Model:        r.Model,
		ParentModel:  r.ParentModel,
		Prompt:       r.Prompt,
		Messages:     messages,
		WordWrap:     r.WordWrap,
		Format:       r.Format,
		System:       r.System,
		Images:       images,
		Options:      opts,
		MultiModal:   r.MultiModal,
		KeepAlive:    r.KeepAlive,
		Think:        think,
		HideThinking: r.HideThinking,
		ShowConnect:  r.ShowConnect,
	}
}

type displayResponseState struct {
	lineLength int
	wordBuffer string
}

func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if runewidth.StringWidth(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				// backtrack the length of the last word and clear to the end of the line
				a := runewidth.StringWidth(state.wordBuffer)
				if a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Printf("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)
				chWidth := runewidth.RuneWidth(ch)

				state.lineLength = runewidth.StringWidth(state.wordBuffer) + chWidth
			} else {
				fmt.Print(string(ch))
				state.lineLength += runewidth.RuneWidth(ch)
				if runewidth.RuneWidth(ch) >= 2 {
					state.wordBuffer = ""
					continue
				}

				switch ch {
				case ' ', '\t':
					state.wordBuffer = ""
				case '\n', '\r':
					state.lineLength = 0
					state.wordBuffer = ""
				default:
					state.wordBuffer += string(ch)
				}
			}
		}
	} else {
		fmt.Printf("%s%s", state.wordBuffer, content)
		if len(state.wordBuffer) > 0 {
			state.wordBuffer = ""
		}
	}
}

func thinkingOutputOpeningText(plainText bool) string {
	text := "Thinking...\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault + readline.ColorGrey
}

func thinkingOutputClosingText(plainText bool) string {
	text := "...done thinking.\n\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault
}

func chat(cmd *cobra.Command, opts runOptions) (*api.Message, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	cancelCtx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var latest api.ChatResponse
	var fullResponse strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false

	role := "assistant"

	fn := func(response api.ChatResponse) error {
		if response.Message.Content != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		latest = response

		role = response.Message.Role
		if response.Message.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(false))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Message.Thinking)
			displayResponse(response.Message.Thinking, opts.WordWrap, state)
		}

		content := response.Message.Content
		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.Message.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(false))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}
		// purposefully not putting thinking blocks in the response, which would
		// only be needed if we later added tool calling to the cli (they get
		// filtered out anyway since current models don't expect them unless you're
		// about to finish some tool calls)
		fullResponse.WriteString(content)

		if response.Message.ToolCalls != nil {
			toolCalls := response.Message.ToolCalls
			if len(toolCalls) > 0 {
				fmt.Print(renderToolCalls(toolCalls, false))
			}
		}

		displayResponse(content, opts.WordWrap, state)

		return nil
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	req := &api.ChatRequest{
		Model:    opts.Model,
		Messages: opts.Messages,
		Format:   json.RawMessage(opts.Format),
		Options:  opts.Options,
		Think:    opts.Think,
	}

	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}

	if err := client.Chat(cancelCtx, req, fn); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, nil
		}

		// this error should ideally be wrapped properly by the client
		if strings.Contains(err.Error(), "upstream error") {
			p.StopAndClear()
			fmt.Println("An error occurred while processing your message. Please try again.")
			fmt.Println()
			return nil, nil
		}
		return nil, err
	}

	if len(opts.Messages) > 0 {
		fmt.Println()
		fmt.Println()
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return nil, err
	}

	if verbose {
		latest.Summary()
	}

	return &api.Message{Role: role, Thinking: thinkingContent.String(), Content: fullResponse.String()}, nil
}

func generate(cmd *cobra.Command, opts runOptions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var latest api.GenerateResponse

	generateContext, ok := cmd.Context().Value(generateContextKey("context")).([]int)
	if !ok {
		generateContext = []int{}
	}

	ctx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false

	plainText := !term.IsTerminal(int(os.Stdout.Fd()))

	fn := func(response api.GenerateResponse) error {
		latest = response
		content := response.Response

		if response.Response != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		if response.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(plainText))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Thinking)
			displayResponse(response.Thinking, opts.WordWrap, state)
		}

		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(plainText))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}

		displayResponse(content, opts.WordWrap, state)

		if response.ToolCalls != nil {
			toolCalls := response.ToolCalls
			if len(toolCalls) > 0 {
				fmt.Print(renderToolCalls(toolCalls, plainText))
			}
		}

		return nil
	}

	if opts.MultiModal {
		opts.Prompt, opts.Images, err = extractFileData(opts.Prompt)
		if err != nil {
			return err
		}
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	request := api.GenerateRequest{
		Model:     opts.Model,
		Prompt:    opts.Prompt,
		Context:   generateContext,
		Images:    opts.Images,
		Format:    json.RawMessage(opts.Format),
		System:    opts.System,
		Options:   opts.Options,
		KeepAlive: opts.KeepAlive,
		Think:     opts.Think,
	}

	if err := client.Generate(ctx, &request, fn); err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		return err
	}

	if opts.Prompt != "" {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done {
		return nil
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return err
	}

	if verbose {
		latest.Summary()
	}

	ctx = context.WithValue(cmd.Context(), generateContextKey("context"), latest.Context)
	cmd.SetContext(ctx)

	return nil
}

func RunServer(_ *cobra.Command, _ []string) error {
	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", envconfig.Host().Host)
	if err != nil {
		return err
	}

	err = server.Serve(ln)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}

	return err
}

func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
		cryptoPublicKey, cryptoPrivateKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privateKeyBytes, err := ssh.MarshalPrivateKey(cryptoPrivateKey, "")
		if err != nil {
			return err
		}

		if err := os.MkdirAll(filepath.Dir(privKeyPath), 0o755); err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		if err := os.WriteFile(privKeyPath, pem.EncodeToMemory(privateKeyBytes), 0o600); err != nil {
			return err
		}

		sshPublicKey, err := ssh.NewPublicKey(cryptoPublicKey)
		if err != nil {
			return err
		}

		publicKeyBytes := ssh.MarshalAuthorizedKey(sshPublicKey)

		if err := os.WriteFile(pubKeyPath, publicKeyBytes, 0o644); err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", publicKeyBytes)
	}
	return nil
}

func checkServerHeartbeat(cmd *cobra.Command, _ []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(cmd.Context()); err != nil {
		if !(strings.Contains(err.Error(), " refused") || strings.Contains(err.Error(), "could not connect")) {
			return err
		}
		if err := startApp(cmd.Context(), client); err != nil {
			return fmt.Errorf("ollama server not responding - %w", err)
		}
	}
	return nil
}

func versionHandler(cmd *cobra.Command, _ []string) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}

	serverVersion, err := client.Version(cmd.Context())
	if err != nil {
		fmt.Println("Warning: could not connect to a running Ollama instance")
	}

	if serverVersion != "" {
		fmt.Printf("ollama version is %s\n", serverVersion)
	}

	if serverVersion != version.Version {
		fmt.Printf("Warning: client version is %s\n", version.Version)
	}
}

func appendEnvDocs(cmd *cobra.Command, envs []envconfig.EnvVar) {
	if len(envs) == 0 {
		return
	}

	envUsage := `
Environment Variables:
`
	for _, e := range envs {
		envUsage += fmt.Sprintf("      %-24s   %s\n", e.Name, e.Description)
	}

	cmd.SetUsageTemplate(cmd.UsageTemplate() + envUsage)
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cobra.EnableCommandSorting = false

	if runtime.GOOS == "windows" && term.IsTerminal(int(os.Stdout.Fd())) {
		console.ConsoleFromFile(os.Stdin) //nolint:errcheck
	}

	rootCmd := &cobra.Command{
		Use:           "ollama",
		Short:         "Large language model runner",
		SilenceUsage:  true,
		SilenceErrors: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		Run: func(cmd *cobra.Command, args []string) {
			if version, _ := cmd.Flags().GetBool("version"); version {
				versionHandler(cmd, args)
				return
			}

			cmd.Print(cmd.UsageString())
		},
	}

	rootCmd.Flags().BoolP("version", "v", false, "Show version information")

	createCmd := &cobra.Command{
		Use:   "create MODEL",
		Short: "Create a model",
		Args:  cobra.ExactArgs(1),
		PreRunE: func(cmd *cobra.Command, args []string) error {
			// Skip server check for experimental mode (writes directly to disk)
			if experimental, _ := cmd.Flags().GetBool("experimental"); experimental {
				return nil
			}
			return checkServerHeartbeat(cmd, args)
		},
		RunE: CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "", "Name of the Modelfile (default \"Modelfile\")")
	createCmd.Flags().StringP("quantize", "q", "", "Quantize model to this level (e.g. q4_K_M)")
	createCmd.Flags().Bool("experimental", false, "Enable experimental safetensors model creation")

	showCmd := &cobra.Command{
		Use:     "show MODEL",
		Short:   "Show information for a model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    ShowHandler,
	}

	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system message of a model")
	showCmd.Flags().BoolP("verbose", "v", false, "Show detailed model information")

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().String("keepalive", "", "Duration to keep a model loaded (e.g. 5m)")
	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")
	runCmd.Flags().String("format", "", "Response format (e.g. json)")
	runCmd.Flags().String("think", "", "Enable thinking mode: true/false or high/medium/low for supported models")
	runCmd.Flags().Lookup("think").NoOptDefVal = "true"
	runCmd.Flags().Bool("hidethinking", false, "Hide thinking output (if provided)")
	runCmd.Flags().Bool("truncate", false, "For embedding models: truncate inputs exceeding context length (default: true). Set --truncate=false to error instead")
	runCmd.Flags().Int("dimensions", 0, "Truncate output embeddings to specified dimension (embedding models only)")
	runCmd.Flags().Bool("experimental", false, "Enable experimental agent loop with tools")
	runCmd.Flags().Bool("experimental-yolo", false, "Skip all tool approval prompts (use with caution)")
	runCmd.Flags().Bool("experimental-websearch", false, "Enable web search tool in experimental mode")

	// Image generation flags (width, height, steps, seed, etc.)
	imagegen.RegisterFlags(runCmd)

	stopCmd := &cobra.Command{
		Use:     "stop MODEL",
		Short:   "Stop a running model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    StopHandler,
	}

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		Args:    cobra.ExactArgs(0),
		RunE:    RunServer,
	}

	pullCmd := &cobra.Command{
		Use:     "pull MODEL",
		Short:   "Pull a model from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:     "push MODEL",
		Short:   "Push a model to a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PushHandler,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	signinCmd := &cobra.Command{
		Use:     "signin",
		Short:   "Sign in to ollama.com",
		Args:    cobra.ExactArgs(0),
		PreRunE: checkServerHeartbeat,
		RunE:    SigninHandler,
	}

	signoutCmd := &cobra.Command{
		Use:     "signout",
		Short:   "Sign out from ollama.com",
		Args:    cobra.ExactArgs(0),
		PreRunE: checkServerHeartbeat,
		RunE:    SignoutHandler,
	}

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		PreRunE: checkServerHeartbeat,
		RunE:    ListHandler,
	}

	psCmd := &cobra.Command{
		Use:     "ps",
		Short:   "List running models",
		PreRunE: checkServerHeartbeat,
		RunE:    ListRunningHandler,
	}
	copyCmd := &cobra.Command{
		Use:     "cp SOURCE DESTINATION",
		Short:   "Copy a model",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    CopyHandler,
	}

	deleteCmd := &cobra.Command{
		Use:     "rm MODEL [MODEL...]",
		Short:   "Remove a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    DeleteHandler,
	}

	runnerCmd := &cobra.Command{
		Use:    "runner",
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runner.Execute(os.Args[1:])
		},
		FParseErrWhitelist: cobra.FParseErrWhitelist{UnknownFlags: true},
	}
	runnerCmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		_ = runner.Execute(args[1:])
	})

	envVars := envconfig.AsMap()

	envs := []envconfig.EnvVar{envVars["OLLAMA_HOST"]}

	for _, cmd := range []*cobra.Command{
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		serveCmd,
	} {
		switch cmd {
		case runCmd:
			imagegen.AppendFlagsDocs(cmd)
			appendEnvDocs(cmd, []envconfig.EnvVar{envVars["OLLAMA_HOST"], envVars["OLLAMA_NOHISTORY"]})
		case serveCmd:
			appendEnvDocs(cmd, []envconfig.EnvVar{
				envVars["OLLAMA_DEBUG"],
				envVars["OLLAMA_HOST"],
				envVars["OLLAMA_CONTEXT_LENGTH"],
				envVars["OLLAMA_KEEP_ALIVE"],
				envVars["OLLAMA_MAX_LOADED_MODELS"],
				envVars["OLLAMA_MAX_QUEUE"],
				envVars["OLLAMA_MODELS"],
				envVars["OLLAMA_NUM_PARALLEL"],
				envVars["OLLAMA_NOPRUNE"],
				envVars["OLLAMA_ORIGINS"],
				envVars["OLLAMA_SCHED_SPREAD"],
				envVars["OLLAMA_FLASH_ATTENTION"],
				envVars["OLLAMA_KV_CACHE_TYPE"],
				envVars["OLLAMA_LLM_LIBRARY"],
				envVars["OLLAMA_GPU_OVERHEAD"],
				envVars["OLLAMA_LOAD_TIMEOUT"],
			})
		default:
			appendEnvDocs(cmd, envs)
		}
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		stopCmd,
		pullCmd,
		pushCmd,
		signinCmd,
		signoutCmd,
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		runnerCmd,
	)

	return rootCmd
}

// If the user has explicitly set thinking options, either through the CLI or
// through the `/set think` or `set nothink` interactive options, then we
// respect them. Otherwise, we check model capabilities to see if the model
// supports thinking. If the model does support thinking, we enable it.
// Otherwise, we unset the thinking option (which is different than setting it
// to false).
//
// If capabilities are not provided, we fetch them from the server.
func inferThinkingOption(caps *[]model.Capability, runOpts *runOptions, explicitlySetByUser bool) (*api.ThinkValue, error) {
	if explicitlySetByUser {
		return runOpts.Think, nil
	}

	if caps == nil {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			return nil, err
		}
		ret, err := client.Show(context.Background(), &api.ShowRequest{
			Model: runOpts.Model,
		})
		if err != nil {
			return nil, err
		}
		caps = &ret.Capabilities
	}

	thinkingSupported := false
	for _, cap := range *caps {
		if cap == model.CapabilityThinking {
			thinkingSupported = true
		}
	}

	if thinkingSupported {
		return &api.ThinkValue{Value: true}, nil
	}

	return nil, nil
}

func renderToolCalls(toolCalls []api.ToolCall, plainText bool) string {
	out := ""
	formatExplanation := ""
	formatValues := ""
	if !plainText {
		formatExplanation = readline.ColorGrey + readline.ColorBold
		formatValues = readline.ColorDefault
		out += formatExplanation
	}
	for i, toolCall := range toolCalls {
		argsAsJSON, err := json.Marshal(toolCall.Function.Arguments)
		if err != nil {
			return ""
		}
		if i > 0 {
			out += "\n"
		}
		// all tool calls are unexpected since we don't currently support registering any in the CLI
		out += fmt.Sprintf("  Model called a non-existent function '%s()' with arguments: %s", formatValues+toolCall.Function.Name+formatExplanation, formatValues+string(argsAsJSON)+formatExplanation)
	}
	if !plainText {
		out += readline.ColorDefault
	}
	return out
}

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
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llama/runner"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

var errModelfileNotFound = errors.New("specified Modelfile wasn't found")

func getModelfileName(cmd *cobra.Command) (string, error) {
	fn, _ := cmd.Flags().GetString("file")

	filename := fn
	if filename == "" {
		filename = "Modelfile"
	}

	absName, err := filepath.Abs(filename)
	if err != nil {
		return "", err
	}

	_, err = os.Stat(absName)
	if err != nil {
		return fn, err
	}

	return absName, nil
}

func CreateHandler(cmd *cobra.Command, args []string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	var reader io.Reader

	filename, err := getModelfileName(cmd)
	if os.IsNotExist(err) {
		if filename == "" {
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

	req, err := modelfile.CreateRequest()
	if err != nil {
		return err
	}
	spinner.Stop()

	req.Name = args[0]
	quantize, _ := cmd.Flags().GetString("quantize")
	if quantize != "" {
		req.Quantize = quantize
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	if len(req.Files) > 0 {
		fileMap := map[string]string{}
		for f, digest := range req.Files {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}
			fileMap[filepath.Base(f)] = digest
		}
		req.Files = fileMap
	}

	if len(req.Adapters) > 0 {
		fileMap := map[string]string{}
		for f, digest := range req.Adapters {
			if _, err := createBlob(cmd, client, f, digest, p); err != nil {
				return err
			}
			fileMap[filepath.Base(f)] = digest
		}
		req.Adapters = fileMap
	}

	bars := make(map[string]*progress.Bar)
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
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

	if err = client.CreateBlob(cmd.Context(), digest, io.TeeReader(bin, &pw)); err != nil {
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

	req := &api.GenerateRequest{
		Model:     opts.Model,
		KeepAlive: opts.KeepAlive,
	}

	return client.Generate(cmd.Context(), req, func(api.GenerateResponse) error { return nil })
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
	}
	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	interactive := true

	opts := runOptions{
		Model:    args[0],
		WordWrap: os.Getenv("TERM") == "xterm-256color",
		Options:  map[string]interface{}{},
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

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

		prompts = append([]string{string(in)}, prompts...)
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

	opts.MultiModal = len(info.ProjectorInfo) != 0
	opts.ParentModel = info.Details.ParentModel

	if interactive {
		if err := loadOrUnloadModel(cmd, &opts); err != nil {
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

		return generateInteractive(cmd, opts)
	}
	return generate(cmd, opts)
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
				bar = progress.NewBar(fmt.Sprintf("pushing %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
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

	n := model.ParseName(args[0])
	if err := client.Push(cmd.Context(), &request, fn); err != nil {
		if spinner != nil {
			spinner.Stop()
		}
		if strings.Contains(err.Error(), "access denied") {
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
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), format.HumanTime(m.ModifiedAt, "Never")})
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
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), procStr, until})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"})
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

	// Unload the model if it's running before deletion
	opts := &runOptions{
		Model:     args[0],
		KeepAlive: &api.Duration{Duration: 0},
	}
	if err := loadOrUnloadModel(cmd, opts); err != nil {
		if !strings.Contains(err.Error(), "not found") {
			return fmt.Errorf("unable to stop existing running model \"%s\": %s", args[0], err)
		}
	}

	for _, name := range args {
		req := api.DeleteRequest{Name: name}
		if err := client.Delete(cmd.Context(), &req); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", name)
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

	for _, boolErr := range []error{errLicense, errModelfile, errParams, errSystem, errTemplate} {
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

	req := api.ShowRequest{Name: args[0]}
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

	return showInfo(resp, os.Stdout)
}

func showInfo(resp *api.ShowResponse, w io.Writer) error {
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
		if resp.ModelInfo != nil {
			arch := resp.ModelInfo["general.architecture"].(string)
			rows = append(rows, []string{"", "architecture", arch})
			rows = append(rows, []string{"", "parameters", format.HumanNumber(uint64(resp.ModelInfo["general.parameter_count"].(float64)))})
			rows = append(rows, []string{"", "context length", strconv.FormatFloat(resp.ModelInfo[fmt.Sprintf("%s.context_length", arch)].(float64), 'f', -1, 64)})
			rows = append(rows, []string{"", "embedding length", strconv.FormatFloat(resp.ModelInfo[fmt.Sprintf("%s.embedding_length", arch)].(float64), 'f', -1, 64)})
		} else {
			rows = append(rows, []string{"", "architecture", resp.Details.Family})
			rows = append(rows, []string{"", "parameters", resp.Details.ParameterSize})
		}
		rows = append(rows, []string{"", "quantization", resp.Details.QuantizationLevel})
		return
	})

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

	head := func(s string, n int) (rows [][]string) {
		scanner := bufio.NewScanner(strings.NewReader(s))
		for scanner.Scan() && (len(rows) < n || n < 0) {
			if text := scanner.Text(); text != "" {
				rows = append(rows, []string{"", strings.TrimSpace(text)})
			}
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
			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				bar = progress.NewBar(fmt.Sprintf("pulling %s...", resp.Digest[7:19]), resp.Total, resp.Completed)
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
	if err := client.Pull(cmd.Context(), &request, fn); err != nil {
		return err
	}

	return nil
}

type generateContextKey string

type runOptions struct {
	Model       string
	ParentModel string
	Prompt      string
	Messages    []api.Message
	WordWrap    bool
	Format      string
	System      string
	Images      []api.ImageData
	Options     map[string]interface{}
	MultiModal  bool
	KeepAlive   *api.Duration
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
				case ' ':
					state.wordBuffer = ""
				case '\n':
					state.lineLength = 0
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
	var latest api.ChatResponse
	var fullResponse strings.Builder
	var role string

	fn := func(response api.ChatResponse) error {
		p.StopAndClear()

		latest = response

		role = response.Message.Role
		content := response.Message.Content
		fullResponse.WriteString(content)

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
	}

	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}

	if err := client.Chat(cancelCtx, req, fn); err != nil {
		if errors.Is(err, context.Canceled) {
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

	return &api.Message{Role: role, Content: fullResponse.String()}, nil
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

	fn := func(response api.GenerateResponse) error {
		p.StopAndClear()

		latest = response
		content := response.Response

		displayResponse(content, opts.WordWrap, state)

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
		if !strings.Contains(err.Error(), " refused") {
			return err
		}
		if err := startApp(cmd.Context(), client); err != nil {
			return errors.New("could not connect to ollama app, is it running?")
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
		Use:     "create MODEL",
		Short:   "Create a model from a Modelfile",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "", "Name of the Modelfile (default \"Modelfile\"")
	createCmd.Flags().StringP("quantize", "q", "", "Quantize model to this level (e.g. q4_0)")

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
	runCmd.Flags().String("format", "", `Response format ("json" or a JSON Schema)`)

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
		Short:  llama.PrintSystemInfo(),
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
			appendEnvDocs(cmd, []envconfig.EnvVar{envVars["OLLAMA_HOST"], envVars["OLLAMA_NOHISTORY"]})
		case serveCmd:
			appendEnvDocs(cmd, []envconfig.EnvVar{
				envVars["OLLAMA_DEBUG"],
				envVars["OLLAMA_HOST"],
				envVars["OLLAMA_KEEP_ALIVE"],
				envVars["OLLAMA_MAX_LOADED_MODELS"],
				envVars["OLLAMA_MAX_QUEUE"],
				envVars["OLLAMA_MODELS"],
				envVars["OLLAMA_NUM_PARALLEL"],
				envVars["OLLAMA_NOPRUNE"],
				envVars["OLLAMA_ORIGINS"],
				envVars["OLLAMA_SCHED_SPREAD"],
				envVars["OLLAMA_TMPDIR"],
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
		listCmd,
		psCmd,
		copyCmd,
		deleteCmd,
		runnerCmd,
	)

	return rootCmd
}

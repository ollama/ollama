package cmd

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
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
)

// ensureThinkingSupport emits a warning if the model does not advertise thinking support
func ensureThinkingSupport(ctx context.Context, client *api.Client, name string) {
	if name == "" {
		return
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		return
	}
	for _, cap := range resp.Capabilities {
		if cap == model.CapabilityThinking {
			return
		}
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

	req, err := modelfile.CreateRequest(filepath.Dir(filename))
	if err != nil {
		return err
	}
	spinner.Stop()

	req.Model = args[0]
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

	req := &api.GenerateRequest{
		Model:     opts.Model,
		KeepAlive: opts.KeepAlive,

		// pass Think here so we fail before getting to the chat prompt if the model doesn't support it
		Think: opts.Think,
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
		return err
	}
	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	interactive := true

	opts := runOptions{
		Model:    args[0],
		WordWrap: os.Getenv("TERM") == "xterm-256color",
		Options:  map[string]any{},
	}

	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	opts.Format = format

	thinkFlag := cmd.Flags().Lookup("think")
	if thinkFlag.Changed {
		think, err := cmd.Flags().GetBool("think")
		if err != nil {
			return err
		}
		opts.Think = &think
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

// showPushSignatureSummary displays post-push signature verification information
func showPushSignatureSummary(modelName model.Name) error {
	sigInfo, err := server.GetSignatureInfo(modelName)
	if err != nil {
		return err
	}

	fmt.Println() // Add spacing after push progress

	if sigInfo == nil {
		// Model is unsigned
		fmt.Printf("📤 Pushed unsigned model %s\n", modelName.DisplayShortest())
		fmt.Printf("   Consider signing the model to ensure authenticity for users\n")
		fmt.Printf("   Use 'ollama sign %s' to add a signature\n", modelName.DisplayShortest())
		return nil
	}

	// Model has signature information
	fmt.Printf("🔐 Pushed signed model %s\n", modelName.DisplayShortest())
	
	// Perform verification to get current status
	verifier := server.NewSignatureVerifier()
	result, err := verifier.VerifyModel(modelName)
	if err != nil {
		fmt.Printf("   ❌ Signature verification failed: %v\n", err)
		fmt.Printf("   Users may see signature verification warnings\n")
		return nil
	}

	if result.Valid {
		fmt.Printf("   ✅ Signature verified successfully\n")
		fmt.Printf("   Users will see verified signature when downloading\n")
		fmt.Printf("   Signer: %s\n", sigInfo.Signer)
		fmt.Printf("   Signed at: %s\n", sigInfo.SignedAt.Format("2006-01-02 15:04:05"))
	} else {
		fmt.Printf("   ❌ Signature verification failed\n")
		if result.ErrorMessage != "" {
			fmt.Printf("   Reason: %s\n", result.ErrorMessage)
		}
		fmt.Printf("   Users will see signature verification warnings\n")
	}

	return nil
}

// performPrePushSignatureChecks validates signature requirements before pushing
func performPrePushSignatureChecks(modelName model.Name, requireSig, verifySig bool) error {
	sigInfo, err := server.GetSignatureInfo(modelName)
	if err != nil {
		return fmt.Errorf("failed to get signature information: %w", err)
	}

	// Check if signature is required but model is unsigned
	if requireSig && sigInfo == nil {
		fmt.Printf("❌ Push failed: Model %s is unsigned\n", modelName.DisplayShortest())
		fmt.Printf("   Use --require-signature=false to push unsigned models\n")
		fmt.Printf("   Or sign the model first: ollama sign %s\n", modelName.DisplayShortest())
		return errors.New("signature required but model is unsigned")
	}

	// If model has signature and verification is enabled, verify it
	if sigInfo != nil && verifySig {
		fmt.Printf("🔍 Verifying signature before push...\n")
		verifier := server.NewSignatureVerifier()
		result, err := verifier.VerifyModel(modelName)
		if err != nil {
			fmt.Printf("❌ Signature verification failed: %v\n", err)
			fmt.Printf("   Use --verify-signature=false to skip verification\n")
			fmt.Printf("   Or fix signature issues first\n")
			return fmt.Errorf("signature verification failed: %w", err)
		}

		if !result.Valid {
			fmt.Printf("❌ Signature is invalid: %s\n", result.ErrorMessage)
			fmt.Printf("   Use --verify-signature=false to push with invalid signature\n")
			fmt.Printf("   Or re-sign the model: ollama sign %s\n", modelName.DisplayShortest())
			return errors.New("signature is invalid")
		}

		fmt.Printf("✅ Signature verified successfully\n")
		fmt.Printf("   Signer: %s\n", sigInfo.Signer)
		fmt.Printf("   Signed: %s\n", sigInfo.SignedAt.Format("2006-01-02 15:04:05"))
	}

	// Show what will be pushed
	if sigInfo != nil {
		fmt.Printf("📦 Ready to push signed model %s\n", modelName.DisplayShortest())
	} else {
		fmt.Printf("📦 Ready to push unsigned model %s\n", modelName.DisplayShortest())
		fmt.Printf("   Consider signing for better security: ollama sign %s\n", modelName.DisplayShortest())
	}

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

	requireSig, err := cmd.Flags().GetBool("require-signature")
	if err != nil {
		return err
	}

	verifySig, err := cmd.Flags().GetBool("verify-signature")
	if err != nil {
		return err
	}

	// Pre-push signature verification and validation
	modelName := model.ParseName(args[0])
	if err := performPrePushSignatureChecks(modelName, requireSig, verifySig); err != nil {
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

	// Show signature verification summary after push completes
	if err := showPushSignatureSummary(n); err != nil {
		// Don't fail the push if we can't show signature info
		slog.Debug("failed to show push signature summary", "error", err)
	}

	destination := n.String()
	if strings.HasSuffix(n.Host, ".ollama.ai") || strings.HasSuffix(n.Host, ".ollama.com") {
		destination = "https://ollama.com/" + strings.TrimSuffix(n.DisplayShortest(), ":latest")
	}
	fmt.Printf("\nYou can find your model at:\n\n")
	fmt.Printf("\t%s\n", destination)

	return nil
}

// formatSignatureStatus converts signature status to a display string
func formatSignatureStatus(sig *api.SignatureStatus) string {
	if sig == nil || !sig.Signed {
		return "Unsigned"
	}
	
	if sig.Verified {
		return "✅ Verified"
	}
	
	return "❌ Invalid"
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
			sigStatus := formatSignatureStatus(m.Signature)
			data = append(data, []string{m.Name, m.Digest[:12], format.HumanBytes(m.Size), format.HumanTime(m.ModifiedAt, "Never"), sigStatus})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED", "SIGNATURE"})
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

// showSignatureVerificationSummary displays post-pull signature verification information and warnings
func showSignatureVerificationSummary(modelName model.Name) error {
	sigInfo, err := server.GetSignatureInfo(modelName)
	if err != nil {
		return err
	}

	fmt.Println() // Add spacing after pull progress

	if sigInfo == nil {
		// Model is unsigned
		fmt.Printf("⚠️  Model %s is unsigned\n", modelName.DisplayShortest())
		fmt.Printf("   Consider verifying the model source and authenticity\n")
		fmt.Printf("   Use 'ollama verify %s' to check signature status\n", modelName.DisplayShortest())
		return nil
	}

	// Model has signature information
	fmt.Printf("🔐 Model %s has signature information\n", modelName.DisplayShortest())
	
	// Perform verification to get current status
	verifier := server.NewSignatureVerifier()
	result, err := verifier.VerifyModel(modelName)
	if err != nil {
		fmt.Printf("   ❌ Signature verification failed: %v\n", err)
		fmt.Printf("   Use 'ollama verify %s -v' for detailed information\n", modelName.DisplayShortest())
		return nil
	}

	if result.Valid {
		fmt.Printf("   ✅ Signature verified successfully\n")
		fmt.Printf("   Signer: %s\n", sigInfo.Signer)
		fmt.Printf("   Signed at: %s\n", sigInfo.SignedAt.Format("2006-01-02 15:04:05"))
	} else {
		fmt.Printf("   ❌ Signature verification failed\n")
		if result.ErrorMessage != "" {
			fmt.Printf("   Reason: %s\n", result.ErrorMessage)
		}
		fmt.Printf("   Use 'ollama verify %s -v' for detailed information\n", modelName.DisplayShortest())
	}

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
	err = client.Pull(cmd.Context(), &request, fn)
	if err != nil {
		return err
	}

	// Show signature verification summary after pull completes
	modelName := model.ParseName(args[0])
	if modelName.IsValid() {
		if err := showSignatureVerificationSummary(modelName); err != nil {
			// Don't fail the pull if we can't show signature info
			slog.Debug("failed to show signature verification summary", "error", err)
		}
	}

	return nil
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
	Think        *bool
	HideThinking bool
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
			}
			displayResponse(response.Message.Thinking, opts.WordWrap, state)
		}

		content := response.Message.Content
		if thinkTagOpened && !thinkTagClosed && content != "" {
			fmt.Print(thinkingOutputClosingText(false))
			thinkTagClosed = true
		}
		// purposefully not putting thinking blocks in the response, which would
		// only be needed if we later added tool calling to the cli (they get
		// filtered out anyway since current models don't expect them unless you're
		// about to finish some tool calls)
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
			}
			displayResponse(response.Thinking, opts.WordWrap, state)
		}

		if thinkTagOpened && !thinkTagClosed && content != "" {
			fmt.Print(thinkingOutputClosingText(plainText))
			thinkTagClosed = true
		}

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

func SignHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return errors.New("sign requires exactly one model name")
	}

	modelName := args[0]
	name := model.ParseName(modelName)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	// Validate the model exists by attempting to parse its manifest
	manifest, err := server.ParseNamedManifest(name)
	if err != nil {
		return fmt.Errorf("model not found: %w", err)
	}

	// Get overwrite flag
	overwrite, _ := cmd.Flags().GetBool("overwrite")

	// Check if model is already signed
	if manifest.Signature != nil {
		fmt.Printf("Model %s is already signed\n", modelName)
		fmt.Printf("  Signer: %s\n", manifest.Signature.Signer)
		fmt.Printf("  Signed at: %s\n", manifest.Signature.SignedAt.Format(time.RFC3339))
		fmt.Printf("  Format: %s\n", manifest.Signature.Format)
		fmt.Printf("  Verified: %v\n", manifest.Signature.Verified)
		
		if !overwrite {
			fmt.Printf("\nUse --overwrite to replace the existing signature\n")
			return nil
		}
		fmt.Printf("\nOverwriting existing signature...\n")
	}

	// Display what would be signed
	fmt.Printf("Model %s found and ready to sign\n", modelName)
	fmt.Printf("  Layers: %d\n", len(manifest.Layers))
	fmt.Printf("  Total size: %s\n", format.HumanBytes(manifest.Size()))
	fmt.Printf("  Digest: %s\n", manifest.Digest()[:12])

	// Check for key/signer options
	keyPath, _ := cmd.Flags().GetString("key")
	sigstoreMode, _ := cmd.Flags().GetBool("sigstore")
	identity, _ := cmd.Flags().GetString("identity")

	if keyPath != "" {
		fmt.Printf("  Signing method: Private key (%s)\n", keyPath)
		return handleKeyBasedSigning(modelName, manifest, keyPath, overwrite)
	} else if sigstoreMode {
		if identity == "" {
			fmt.Printf("  Signing method: Sigstore (keyless)\n")
			fmt.Printf("  NOTE: Will use OIDC identity from environment\n")
		} else {
			fmt.Printf("  Signing method: Sigstore with identity %s\n", identity)
		}
		fmt.Printf("  NOTE: Sigstore signing will be implemented in a future commit\n")
		fmt.Printf("  Use --key option for basic ed25519 signing\n")
		return nil
	} else {
		fmt.Printf("  No signing method specified - generating test signature\n")
		return handleTestSigning(modelName, manifest, overwrite)
	}

	return nil
}

// handleKeyBasedSigning handles signing with a provided private key file
func handleKeyBasedSigning(modelName string, manifest *server.Manifest, keyPath string, overwrite bool) error {
	// Check if key file exists
	if _, err := os.Stat(keyPath); os.IsNotExist(err) {
		return fmt.Errorf("private key file not found: %s", keyPath)
	}

	// Read private key file (assuming ed25519 private key in base64)
	keyData, err := os.ReadFile(keyPath)
	if err != nil {
		return fmt.Errorf("failed to read private key: %w", err)
	}

	privateKeyB64 := strings.TrimSpace(string(keyData))

	// Compute model digest
	modelDigest, err := server.ComputeModelDigest(manifest)
	if err != nil {
		return fmt.Errorf("failed to compute model digest: %w", err)
	}

	// Create signature
	signer := "user@localhost" // Default signer
	oms, err := server.CreateTestSignature(privateKeyB64, modelDigest, signer)
	if err != nil {
		return fmt.Errorf("failed to create signature: %w", err)
	}

	// Save signature and update manifest
	if err := saveSignatureToModel(modelName, manifest, oms, overwrite); err != nil {
		return fmt.Errorf("failed to save signature: %w", err)
	}

	fmt.Printf("✅ Successfully signed model %s\n", modelName)
	fmt.Printf("   Signer: %s\n", signer)
	fmt.Printf("   Algorithm: %s\n", oms.Algorithm)
	fmt.Printf("   Timestamp: %s\n", oms.Timestamp.Format("2006-01-02 15:04:05"))

	return nil
}

// handleTestSigning generates a test key pair and signs the model
func handleTestSigning(modelName string, manifest *server.Manifest, overwrite bool) error {
	fmt.Printf("  Generating test ed25519 key pair...\n")

	// Generate test key pair
	publicKey, privateKey, err := server.GenerateKeyPair()
	if err != nil {
		return fmt.Errorf("failed to generate key pair: %w", err)
	}

	// Compute model digest
	modelDigest, err := server.ComputeModelDigest(manifest)
	if err != nil {
		return fmt.Errorf("failed to compute model digest: %w", err)
	}

	// Create test signature
	signer := "test-signer@localhost"
	oms, err := server.CreateTestSignature(privateKey, modelDigest, signer)
	if err != nil {
		return fmt.Errorf("failed to create test signature: %w", err)
	}

	// Save signature and update manifest
	if err := saveSignatureToModel(modelName, manifest, oms, overwrite); err != nil {
		return fmt.Errorf("failed to save signature: %w", err)
	}

	fmt.Printf("✅ Successfully signed model %s with test signature\n", modelName)
	fmt.Printf("   Signer: %s\n", signer)
	fmt.Printf("   Algorithm: %s\n", oms.Algorithm)
	fmt.Printf("   Public Key: %s...\n", publicKey[:32])
	fmt.Printf("   Timestamp: %s\n", oms.Timestamp.Format("2006-01-02 15:04:05"))
	fmt.Printf("   NOTE: This is a test signature for development purposes\n")

	return nil
}

// saveSignatureToModel saves the signature to blob storage and updates the manifest
func saveSignatureToModel(modelName string, manifest *server.Manifest, oms *server.OMSSignature, overwrite bool) error {
	// Marshal signature to JSON
	sigData, err := json.MarshalIndent(oms, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal signature: %w", err)
	}

	// Compute signature digest
	sigDigest := "sha256:" + fmt.Sprintf("%x", sha256.Sum256(sigData))

	// Save signature to blob storage
	sigPath, err := server.GetBlobsPath(sigDigest)
	if err != nil {
		return fmt.Errorf("failed to get signature blob path: %w", err)
	}

	if err := os.WriteFile(sigPath, sigData, 0o644); err != nil {
		return fmt.Errorf("failed to write signature blob: %w", err)
	}

	// Create signature layer
	sigLayer := server.Layer{
		MediaType: server.MediaTypeModelSignature,
		Digest:    sigDigest,
		Size:      int64(len(sigData)),
	}

	// Update manifest with signature information
	manifest.AddSignatureLayer(sigLayer)
	manifest.Signature = &server.SignatureInfo{
		Format:       oms.Version,
		SignatureURI: sigDigest,
		Verified:     false, // Will be verified later
		Signer:       oms.Signer,
		SignedAt:     oms.Timestamp,
	}

	// Save updated manifest
	return saveUpdatedManifest(modelName, manifest)
}

// saveUpdatedManifest saves the updated manifest back to disk
func saveUpdatedManifest(modelName string, manifest *server.Manifest) error {
	// Parse model name to get manifest path
	modelPath := model.ParseName(modelName)
	if !modelPath.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	// Get manifest directory
	manifestsDir, err := server.GetManifestPath()
	if err != nil {
		return fmt.Errorf("failed to get manifest path: %w", err)
	}

	// Construct manifest file path
	manifestPath := filepath.Join(manifestsDir, modelPath.Filepath())

	// Marshal manifest
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	// Write manifest
	if err := os.WriteFile(manifestPath, manifestData, 0o644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	return nil
}

// ConfigSignatureHandler manages signature configuration settings
func ConfigSignatureHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		// Show current configuration
		config, err := server.LoadSignatureConfig()
		if err != nil {
			return fmt.Errorf("failed to load signature configuration: %w", err)
		}

		fmt.Printf("Current signature configuration:\n\n")
		fmt.Printf("  Policy: %s\n", config.Policy)
		fmt.Printf("  Verify on pull: %t\n", config.VerifyOnPull)
		fmt.Printf("  Verify on push: %t\n", config.VerifyOnPush)
		fmt.Printf("  Require trusted signers: %t\n", config.RequireTrustedSigners)
		fmt.Printf("  Max signature age: %d days\n", config.MaxSignatureAge)
		fmt.Printf("  Check revocation: %t\n", config.CheckRevocation)
		fmt.Printf("  Trusted signers: %d\n", len(config.TrustedSigners))

		if len(config.TrustedSigners) > 0 {
			fmt.Printf("\nTrusted signers:\n")
			for _, signer := range config.TrustedSigners {
				fmt.Printf("  - %s (%s)\n", signer.Name, signer.Email)
				if signer.Description != "" {
					fmt.Printf("    %s\n", signer.Description)
				}
			}
		}

		fmt.Printf("\nLast updated: %s\n", config.UpdatedAt.Format("2006-01-02 15:04:05"))
		
		fmt.Printf("\nAvailable policies:\n")
		fmt.Printf("  permissive - Allow unsigned models (default)\n")
		fmt.Printf("  warn       - Warn about unsigned models but allow them\n")
		fmt.Printf("  strict     - Require valid signatures for all models\n")

		return nil
	}

	// Handle configuration commands
	subcommand := args[0]
	switch subcommand {
	case "set":
		return handleConfigSet(cmd, args[1:])
	case "add-signer":
		return handleAddSigner(cmd, args[1:])
	case "remove-signer":
		return handleRemoveSigner(cmd, args[1:])
	case "reset":
		return handleConfigReset(cmd)
	default:
		return fmt.Errorf("unknown config subcommand: %s", subcommand)
	}
}

func handleConfigSet(cmd *cobra.Command, args []string) error {
	if len(args) != 2 {
		return errors.New("usage: config set <key> <value>")
	}

	key, value := args[0], args[1]
	config, err := server.LoadSignatureConfig()
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	switch key {
	case "policy":
		switch server.SignaturePolicy(value) {
		case server.PolicyPermissive, server.PolicyWarn, server.PolicyStrict:
			config.Policy = server.SignaturePolicy(value)
		default:
			return fmt.Errorf("invalid policy: %s (must be permissive, warn, or strict)", value)
		}
	case "verify-on-pull":
		config.VerifyOnPull = value == "true"
	case "verify-on-push":
		config.VerifyOnPush = value == "true"
	case "require-trusted-signers":
		config.RequireTrustedSigners = value == "true"
	case "check-revocation":
		config.CheckRevocation = value == "true"
	default:
		return fmt.Errorf("unknown configuration key: %s", key)
	}

	if err := server.SaveSignatureConfig(config); err != nil {
		return fmt.Errorf("failed to save configuration: %w", err)
	}

	fmt.Printf("✅ Configuration updated: %s = %s\n", key, value)
	return nil
}

func handleAddSigner(cmd *cobra.Command, args []string) error {
	if len(args) < 2 {
		return errors.New("usage: config add-signer <name> <email> [public-key] [description]")
	}

	name := args[0]
	email := args[1]
	publicKey := ""
	description := ""

	if len(args) > 2 {
		publicKey = args[2]
	}
	if len(args) > 3 {
		description = args[3]
	}

	config, err := server.LoadSignatureConfig()
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	signer := server.TrustedSigner{
		ID:          fmt.Sprintf("signer-%d", time.Now().Unix()),
		Name:        name,
		Email:       email,
		PublicKey:   publicKey,
		Description: description,
	}

	if err := config.AddTrustedSigner(signer); err != nil {
		return fmt.Errorf("failed to add signer: %w", err)
	}

	if err := server.SaveSignatureConfig(config); err != nil {
		return fmt.Errorf("failed to save configuration: %w", err)
	}

	fmt.Printf("✅ Added trusted signer: %s (%s)\n", name, email)
	return nil
}

func handleRemoveSigner(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return errors.New("usage: config remove-signer <email>")
	}

	email := args[0]
	config, err := server.LoadSignatureConfig()
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	// Find signer by email
	var signerID string
	for _, signer := range config.TrustedSigners {
		if signer.Email == email {
			signerID = signer.ID
			break
		}
	}

	if signerID == "" {
		return fmt.Errorf("signer not found: %s", email)
	}

	if err := config.RemoveTrustedSigner(signerID); err != nil {
		return fmt.Errorf("failed to remove signer: %w", err)
	}

	if err := server.SaveSignatureConfig(config); err != nil {
		return fmt.Errorf("failed to save configuration: %w", err)
	}

	fmt.Printf("✅ Removed trusted signer: %s\n", email)
	return nil
}

func handleConfigReset(cmd *cobra.Command) error {
	config := server.DefaultSignatureConfig()
	if err := server.SaveSignatureConfig(config); err != nil {
		return fmt.Errorf("failed to reset configuration: %w", err)
	}

	fmt.Printf("✅ Configuration reset to defaults\n")
	return nil
}

func VerifyHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return errors.New("verify requires exactly one model name")
	}

	modelName := args[0]
	name := model.ParseName(modelName)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", modelName)
	}

	// Get signature information
	sigInfo, err := server.GetSignatureInfo(name)
	if err != nil {
		return fmt.Errorf("failed to get signature info: %w", err)
	}

	// Check if model is signed
	if sigInfo == nil {
		fmt.Printf("Model %s is not signed\n", modelName)
		fmt.Printf("  Use 'ollama sign %s' to sign this model\n", modelName)
		return nil
	}

	fmt.Printf("Model %s signature information:\n", modelName)
	fmt.Printf("  Signer: %s\n", sigInfo.Signer)
	fmt.Printf("  Format: %s\n", sigInfo.Format)
	fmt.Printf("  Signed at: %s\n", sigInfo.SignedAt.Format(time.RFC3339))
	fmt.Printf("  Signature URI: %s\n", sigInfo.SignatureURI)

	// Perform verification using our signature verifier
	verifier := server.NewSignatureVerifier()
	result, err := verifier.VerifyModel(name)
	if err != nil {
		fmt.Printf("  Status: ❌ Verification failed\n")
		fmt.Printf("  Error: %v\n", err)
		return nil // Don't return error for verification failures
	}

	if result.Valid {
		fmt.Printf("  Status: ✅ Signature verified\n")
		if result.ErrorMessage != "" {
			fmt.Printf("  Note: %s\n", result.ErrorMessage)
		}
	} else {
		fmt.Printf("  Status: ❌ Signature invalid\n")
		if result.ErrorMessage != "" {
			fmt.Printf("  Reason: %s\n", result.ErrorMessage)
		}
	}

	// Show additional verification details if verbose flag is set
	verbose, _ := cmd.Flags().GetBool("verbose")
	if verbose {
		fmt.Printf("\nDetailed verification information:\n")
		fmt.Printf("  Verification format: %s\n", result.Format)
		if result.Signer != "" {
			fmt.Printf("  Verified signer: %s\n", result.Signer)
		}
		if !result.SignedAt.IsZero() {
			fmt.Printf("  Verified signing time: %s\n", result.SignedAt.Format(time.RFC3339))
		}
		
		// Show manifest info
		manifest, err := server.ParseNamedManifest(name)
		if err == nil {
			fmt.Printf("  Model layers: %d\n", len(manifest.Layers))
			fmt.Printf("  Model size: %s\n", format.HumanBytes(manifest.Size()))
			fmt.Printf("  Manifest digest: %s\n", manifest.Digest()[:12])
			
			sigLayer := manifest.GetSignatureLayer()
			if sigLayer != nil {
				fmt.Printf("  Signature layer digest: %s\n", sigLayer.Digest[:12])
				fmt.Printf("  Signature layer size: %s\n", format.HumanBytes(sigLayer.Size))
			}
		}
	}

	return nil
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
		Short:   "Create a model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "", "Name of the Modelfile (default \"Modelfile\")")
	createCmd.Flags().StringP("quantize", "q", "", "Quantize model to this level (e.g. q4_K_M)")

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
	runCmd.Flags().Bool("think", false, "Whether to use thinking mode for supported models")
	runCmd.Flags().Bool("hidethinking", false, "Hide thinking output (if provided)")

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
	pushCmd.Flags().Bool("require-signature", false, "Require model to be signed before pushing")
	pushCmd.Flags().Bool("verify-signature", true, "Verify signature integrity before pushing")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		PreRunE: checkServerHeartbeat,
		RunE:    ListHandler,
	}

	signCmd := &cobra.Command{
		Use:     "sign MODEL",
		Short:   "Sign a model for integrity verification",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    SignHandler,
	}

	signCmd.Flags().String("key", "", "Path to private key file for signing")
	signCmd.Flags().Bool("sigstore", false, "Use Sigstore for keyless signing")
	signCmd.Flags().String("identity", "", "Identity to use for Sigstore signing")
	signCmd.Flags().Bool("overwrite", false, "Overwrite existing signature")

	verifyCmd := &cobra.Command{
		Use:     "verify MODEL",
		Short:   "Verify the signature of a model",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    VerifyHandler,
	}

	verifyCmd.Flags().BoolP("verbose", "v", false, "Show detailed verification information")

	configCmd := &cobra.Command{
		Use:   "config [SUBCOMMAND]",
		Short: "Manage signature verification configuration",
		Long:  "Manage signature verification policies, trusted signers, and verification settings",
		Args:  cobra.ArbitraryArgs,
		RunE:  ConfigSignatureHandler,
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
		signCmd,
		verifyCmd,
		configCmd,
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
func inferThinkingOption(caps *[]model.Capability, runOpts *runOptions, explicitlySetByUser bool) (*bool, error) {
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
		thinking := true
		return &thinking, nil
	}

	return nil, nil
}

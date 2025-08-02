package cmd

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/dustin/go-humanize"
	"github.com/olekukonko/tablewriter"
	"github.com/pdevine/readline"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/progressbar"
	"github.com/jmorganca/ollama/server"
	"github.com/jmorganca/ollama/version"
)

type Painter struct {
	IsMultiLine bool
}

func (p Painter) Paint(line []rune, _ int) []rune {
	termType := os.Getenv("TERM")
	if termType == "xterm-256color" && len(line) == 0 {
		var prompt string
		if p.IsMultiLine {
			prompt = "Use \"\"\" to end multi-line input"
		} else {
			prompt = "Send a message (/? for help)"
		}
		return []rune(fmt.Sprintf("\033[38;5;245m%s\033[%dD\033[0m", prompt, len(prompt)))
	}
	// add a space and a backspace to prevent the cursor from walking up the screen
	line = append(line, []rune(" \b")...)
	return line
}

func CreateHandler(cmd *cobra.Command, args []string) error {
	filename, _ := cmd.Flags().GetString("file")
	filename, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	var spinner *Spinner

	var currentDigest string
	var bar *progressbar.ProgressBar

	request := api.CreateRequest{Name: args[0], Path: filename}
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != currentDigest && resp.Digest != "" {
			if spinner != nil {
				spinner.Stop()
			}
			currentDigest = resp.Digest
			switch {
			case strings.Contains(resp.Status, "embeddings"):
				bar = progressbar.Default(resp.Total, resp.Status)
				bar.Set64(resp.Completed)
			default:
				// pulling
				bar = progressbar.DefaultBytes(
					resp.Total,
					resp.Status,
				)
				bar.Set64(resp.Completed)
			}
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set64(resp.Completed)
		} else {
			currentDigest = ""
			if spinner != nil {
				spinner.Stop()
			}
			spinner = NewSpinner(resp.Status)
			go spinner.Spin(100 * time.Millisecond)
		}

		return nil
	}

	if err := client.Create(context.Background(), &request, fn); err != nil {
		return err
	}

	if spinner != nil {
		spinner.Stop()
		if spinner.description != "success" {
			return errors.New("unexpected end to create model")
		}
	}

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	models, err := client.List(context.Background())
	if err != nil {
		return err
	}

	canonicalModelPath := server.ParseModelPath(args[0])
	for _, model := range models.Models {
		if model.Name == canonicalModelPath.GetShortTagname() {
			return RunGenerate(cmd, args)
		}
	}

	if err := PullHandler(cmd, args); err != nil {
		return err
	}

	return RunGenerate(cmd, args)
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	var currentDigest string
	var bar *progressbar.ProgressBar

	request := api.PushRequest{Name: args[0], Insecure: insecure}
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != currentDigest && resp.Digest != "" {
			currentDigest = resp.Digest
			bar = progressbar.DefaultBytes(
				resp.Total,
				fmt.Sprintf("pushing %s...", resp.Digest[7:19]),
			)

			bar.Set64(resp.Completed)
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set64(resp.Completed)
		} else {
			currentDigest = ""
			fmt.Println(resp.Status)
		}
		return nil
	}

	if err := client.Push(context.Background(), &request, fn); err != nil {
		return err
	}

	if bar != nil && !bar.IsFinished() {
		return errors.New("unexpected end to push model")
	}

	return nil
}

func ListHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	models, err := client.List(context.Background())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			data = append(data, []string{m.Name, m.Digest[:12], humanize.Bytes(uint64(m.Size)), format.HumanTime(m.ModifiedAt, "Never")})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "ID", "SIZE", "MODIFIED"})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetNoWhiteSpace(true)
	table.SetTablePadding("\t")
	table.AppendBulk(data)
	table.Render()

	return nil
}

func DeleteHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	for _, name := range args {
		req := api.DeleteRequest{Name: name}
		if err := client.Delete(context.Background(), &req); err != nil {
			return err
		}
		fmt.Printf("deleted '%s'\n", name)
	}
	return nil
}

func ShowHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	if len(args) != 1 {
		return errors.New("missing model name")
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
	} else if flagsSet == 0 {
		return errors.New("one of '--license', '--modelfile', '--parameters', '--system', or '--template' must be specified")
	}

	req := api.ShowRequest{Name: args[0]}
	resp, err := client.Show(context.Background(), &req)
	if err != nil {
		return err
	}

	switch showType {
	case "license":
		fmt.Println(resp.License)
	case "modelfile":
		fmt.Println(resp.Modelfile)
	case "parameters":
		fmt.Println(resp.Parameters)
	case "system":
		fmt.Println(resp.System)
	case "template":
		fmt.Println(resp.Template)
	}

	return nil
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	req := api.CopyRequest{Source: args[0], Destination: args[1]}
	if err := client.Copy(context.Background(), &req); err != nil {
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

	return pull(args[0], insecure)
}

func pull(model string, insecure bool) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	var currentDigest string
	var bar *progressbar.ProgressBar

	request := api.PullRequest{Name: model, Insecure: insecure}
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != currentDigest && resp.Digest != "" {
			currentDigest = resp.Digest
			bar = progressbar.DefaultBytes(
				resp.Total,
				fmt.Sprintf("pulling %s...", resp.Digest[7:19]),
			)

			bar.Set64(resp.Completed)
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set64(resp.Completed)
		} else {
			currentDigest = ""
			fmt.Println(resp.Status)
		}

		return nil
	}

	if err := client.Pull(context.Background(), &request, fn); err != nil {
		return err
	}

	if bar != nil && !bar.IsFinished() {
		return errors.New("unexpected end to pull model")
	}

	return nil
}

func RunGenerate(cmd *cobra.Command, args []string) error {
	if len(args) > 1 {
		// join all args into a single prompt
		return generate(cmd, args[0], strings.Join(args[1:], " "))
	}

	if readline.IsTerminal(int(os.Stdin.Fd())) {
		return generateInteractive(cmd, args[0])
	}

	return generateBatch(cmd, args[0])
}

type generateContextKey string

func generate(cmd *cobra.Command, model, prompt string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	spinner := NewSpinner("")
	go spinner.Spin(60 * time.Millisecond)

	var latest api.GenerateResponse

	generateContext, ok := cmd.Context().Value(generateContextKey("context")).([]int)
	if !ok {
		generateContext = []int{}
	}

	var wrapTerm bool
	termType := os.Getenv("TERM")
	if termType == "xterm-256color" {
		wrapTerm = true
	}

	termWidth, _, err := term.GetSize(int(0))
	if err != nil {
		wrapTerm = false
	}

	// override wrapping if the user turned it off
	nowrap, err := cmd.Flags().GetBool("nowordwrap")
	if err != nil {
		return err
	}
	if nowrap {
		wrapTerm = false
	}

	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)
	var abort bool

	go func() {
		<-sigChan
		cancel()
		abort = true
	}()

	var currentLineLength int
	var wordBuffer string

	request := api.GenerateRequest{Model: model, Prompt: prompt, Context: generateContext}
	fn := func(response api.GenerateResponse) error {
		if !spinner.IsFinished() {
			spinner.Finish()
		}

		latest = response

		if wrapTerm {
			for _, ch := range response.Response {
				if currentLineLength+1 > termWidth-5 {
					// backtrack the length of the last word and clear to the end of the line
					fmt.Printf("\x1b[%dD\x1b[K\n", len(wordBuffer))
					fmt.Printf("%s%c", wordBuffer, ch)
					currentLineLength = len(wordBuffer) + 1
				} else {
					fmt.Print(string(ch))
					currentLineLength += 1

					switch ch {
					case ' ':
						wordBuffer = ""
					case '\n':
						currentLineLength = 0
					default:
						wordBuffer += string(ch)
					}
				}
			}
		} else {
			fmt.Print(response.Response)
		}

		return nil
	}

	if err := client.Generate(cancelCtx, &request, fn); err != nil {
		if strings.Contains(err.Error(), "failed to load model") {
			// tell the user to check the server log, if it exists locally
			home, nestedErr := os.UserHomeDir()
			if nestedErr != nil {
				// return the original error
				return err
			}
			logPath := filepath.Join(home, ".ollama", "logs", "server.log")
			if _, nestedErr := os.Stat(logPath); nestedErr == nil {
				err = fmt.Errorf("%w\nFor more details, check the error logs at %s", err, logPath)
			}
		} else if strings.Contains(err.Error(), "context canceled") && abort {
			spinner.Finish()
			return nil
		}
		return err
	}
	if prompt != "" {
		fmt.Println()
		fmt.Println()
	}

	if !latest.Done {
		if abort {
			return nil
		}
		return errors.New("unexpected end of response")
	}

	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return err
	}

	if verbose {
		latest.Summary()
	}

	ctx := cmd.Context()
	ctx = context.WithValue(ctx, generateContextKey("context"), latest.Context)
	cmd.SetContext(ctx)

	return nil
}

func generateInteractive(cmd *cobra.Command, model string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	// load the model
	if err := generate(cmd, model, ""); err != nil {
		return err
	}

	completer := readline.NewPrefixCompleter(
		readline.PcItem("/help"),
		readline.PcItem("/list"),
		readline.PcItem("/set",
			readline.PcItem("history"),
			readline.PcItem("nohistory"),
			readline.PcItem("wordwrap"),
			readline.PcItem("nowordwrap"),
			readline.PcItem("verbose"),
			readline.PcItem("quiet"),
		),
		readline.PcItem("/show",
			readline.PcItem("license"),
			readline.PcItem("modelfile"),
			readline.PcItem("parameters"),
			readline.PcItem("system"),
			readline.PcItem("template"),
		),
		readline.PcItem("/exit"),
		readline.PcItem("/bye"),
	)

	usage := func() {
		fmt.Fprintln(os.Stderr, "commands:")
		fmt.Fprintln(os.Stderr, completer.Tree("  "))
	}

	var painter Painter

	config := readline.Config{
		Painter:      &painter,
		Prompt:       ">>> ",
		HistoryFile:  filepath.Join(home, ".ollama", "history"),
		AutoComplete: completer,
	}

	scanner, err := readline.NewEx(&config)
	if err != nil {
		return err
	}
	defer scanner.Close()

	var multiLineBuffer string
	var isMultiLine bool

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("Use Ctrl-D or /bye to exit.")
			}

			continue
		case err != nil:
			return err
		}

		line = strings.TrimSpace(line)

		switch {
		case isMultiLine:
			if strings.HasSuffix(line, `"""`) {
				isMultiLine = false
				painter.IsMultiLine = isMultiLine
				multiLineBuffer += strings.TrimSuffix(line, `"""`)
				line = multiLineBuffer
				multiLineBuffer = ""
				scanner.SetPrompt(">>> ")
			} else {
				multiLineBuffer += line + " "
				continue
			}
		case strings.HasPrefix(line, `"""`):
			isMultiLine = true
			painter.IsMultiLine = isMultiLine
			multiLineBuffer = strings.TrimPrefix(line, `"""`) + " "
			scanner.SetPrompt("... ")
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "history":
					scanner.HistoryEnable()
				case "nohistory":
					scanner.HistoryDisable()
				case "wordwrap":
					cmd.Flags().Set("nowordwrap", "false")
					fmt.Println("Set 'wordwrap' mode.")
				case "nowordwrap":
					cmd.Flags().Set("nowordwrap", "true")
					fmt.Println("Set 'nowordwrap' mode.")
				case "verbose":
					cmd.Flags().Set("verbose", "true")
					fmt.Println("Set 'verbose' mode.")
				case "quiet":
					cmd.Flags().Set("verbose", "false")
					fmt.Println("Set 'quiet' mode.")
				case "mode":
					if len(args) > 2 {
						switch args[2] {
						case "vim":
							scanner.SetVimMode(true)
						case "emacs", "default":
							scanner.SetVimMode(false)
						default:
							usage()
						}
					} else {
						usage()
					}
				default:
					fmt.Printf("Unknown command '/set %s'. Type /? for help\n", args[1])
				}
			} else {
				usage()
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) > 1 {
				resp, err := server.GetModelInfo(model)
				if err != nil {
					fmt.Println("error: couldn't get model")
					return err
				}

				switch args[1] {
				case "license":
					fmt.Println(resp.License)
				case "modelfile":
					fmt.Println(resp.Modelfile)
				case "parameters":
					fmt.Println(resp.Parameters)
				case "system":
					fmt.Println(resp.System)
				case "template":
					fmt.Println(resp.Template)
				default:
					fmt.Printf("Unknown command '/show %s'. Type /? for help\n", args[1])
				}
			} else {
				usage()
			}
		case line == "/help", line == "/?":
			usage()
		case line == "/exit", line == "/bye":
			return nil
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			fmt.Printf("Unknown command '%s'. Type /? for help\n", args[0])
		}

		if len(line) > 0 && line[0] != '/' {
			if err := generate(cmd, model, line); err != nil {
				return err
			}
		}
	}
}

func generateBatch(cmd *cobra.Command, model string) error {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		prompt := scanner.Text()
		fmt.Printf(">>> %s\n", prompt)
		if err := generate(cmd, model, prompt); err != nil {
			return err
		}
	}

	return nil
}

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port, err := net.SplitHostPort(os.Getenv("OLLAMA_HOST"))
	if err != nil {
		host, port = "127.0.0.1", "11434"
		if ip := net.ParseIP(strings.Trim(os.Getenv("OLLAMA_HOST"), "[]")); ip != nil {
			host = ip.String()
		}
	}

	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return err
	}

	var origins []string
	if o := os.Getenv("OLLAMA_ORIGINS"); o != "" {
		origins = strings.Split(o, ",")
	}

	if noprune := os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		if err := server.PruneLayers(); err != nil {
			return err
		}

		manifestsPath, err := server.GetManifestPath()
		if err != nil {
			return err
		}

		if err := server.PruneDirectory(manifestsPath); err != nil {
			return err
		}
	}

	return server.Serve(ln, origins)
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
		_, privKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privKeyBytes, err := format.OpenSSHPrivateKey(privKey, "")
		if err != nil {
			return err
		}

		err = os.MkdirAll(filepath.Dir(privKeyPath), 0o755)
		if err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		err = os.WriteFile(privKeyPath, pem.EncodeToMemory(privKeyBytes), 0o600)
		if err != nil {
			return err
		}

		sshPrivateKey, err := ssh.NewSignerFromKey(privKey)
		if err != nil {
			return err
		}

		pubKeyData := ssh.MarshalAuthorizedKey(sshPrivateKey.PublicKey())

		err = os.WriteFile(pubKeyPath, pubKeyData, 0o644)
		if err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", string(pubKeyData))
	}
	return nil
}

func startMacApp(client *api.Client) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	link, err := os.Readlink(exe)
	if err != nil {
		return err
	}
	if !strings.Contains(link, "Ollama.app") {
		return fmt.Errorf("could not find ollama app")
	}
	path := strings.Split(link, "Ollama.app")
	if err := exec.Command("/usr/bin/open", "-a", path[0]+"Ollama.app").Run(); err != nil {
		return err
	}
	// wait for the server to start
	timeout := time.After(5 * time.Second)
	tick := time.Tick(500 * time.Millisecond)
	for {
		select {
		case <-timeout:
			return errors.New("timed out waiting for server to start")
		case <-tick:
			if err := client.Heartbeat(context.Background()); err == nil {
				return nil // server has started
			}
		}
	}
}

func checkServerHeartbeat(_ *cobra.Command, _ []string) error {
	client, err := api.FromEnv()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(context.Background()); err != nil {
		if !strings.Contains(err.Error(), "connection refused") {
			return err
		}
		if runtime.GOOS == "darwin" {
			if err := startMacApp(client); err != nil {
				return fmt.Errorf("could not connect to ollama app, is it running?")
			}
		} else {
			return fmt.Errorf("could not connect to ollama server, run 'ollama serve' to start it")
		}
	}
	return nil
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	rootCmd := &cobra.Command{
		Use:           "ollama",
		Short:         "Large language model runner",
		SilenceUsage:  true,
		SilenceErrors: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
		Version: version.Version,
	}

	cobra.EnableCommandSorting = false

	createCmd := &cobra.Command{
		Use:     "create MODEL",
		Short:   "Create a model from a Modelfile",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")

	showCmd := &cobra.Command{
		Use:     "show MODEL",
		Short:   "Show information for a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    ShowHandler,
	}

	showCmd.Flags().Bool("license", false, "Show license of a model")
	showCmd.Flags().Bool("modelfile", false, "Show Modelfile of a model")
	showCmd.Flags().Bool("parameters", false, "Show parameters of a model")
	showCmd.Flags().Bool("template", false, "Show template of a model")
	showCmd.Flags().Bool("system", false, "Show system prompt of a model")

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")
	runCmd.Flags().Bool("nowordwrap", false, "Don't wrap words to the next line automatically")

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE:    RunServer,
	}

	pullCmd := &cobra.Command{
		Use:     "pull MODEL",
		Short:   "Pull a model from a registry",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:     "push MODEL",
		Short:   "Push a model to a registry",
		Args:    cobra.MinimumNArgs(1),
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

	copyCmd := &cobra.Command{
		Use:     "cp",
		Short:   "Copy a model",
		Args:    cobra.MinimumNArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    CopyHandler,
	}

	deleteCmd := &cobra.Command{
		Use:     "rm",
		Short:   "Remove a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    DeleteHandler,
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		showCmd,
		runCmd,
		pullCmd,
		pushCmd,
		listCmd,
		copyCmd,
		deleteCmd,
	)

	return rootCmd
}

package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/chzyer/readline"
	"github.com/dustin/go-humanize"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/progressbar"
	"github.com/jmorganca/ollama/server"
)

func CreateHandler(cmd *cobra.Command, args []string) error {
	filename, _ := cmd.Flags().GetString("file")
	filename, err := filepath.Abs(filename)
	if err != nil {
		return err
	}

	client := api.NewClient()

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
				bar = progressbar.Default(int64(resp.Total), resp.Status)
				bar.Set(resp.Completed)
			default:
				// pulling
				bar = progressbar.DefaultBytes(
					int64(resp.Total),
					resp.Status,
				)
				bar.Set(resp.Completed)
			}
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set(resp.Completed)
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
	}

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	mp := server.ParseModelPath(args[0])
	fp, err := mp.GetManifestPath(false)
	if err != nil {
		return err
	}

	_, err = os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := pull(args[0], false); err != nil {
			var apiStatusError api.StatusError
			if !errors.As(err, &apiStatusError) {
				return err
			}

			if apiStatusError.StatusCode != http.StatusBadGateway {
				return err
			}
		}
	case err != nil:
		return err
	}

	return RunGenerate(cmd, args)
}

func PushHandler(cmd *cobra.Command, args []string) error {
	client := api.NewClient()

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
				int64(resp.Total),
				fmt.Sprintf("pushing %s...", resp.Digest[7:19]),
			)

			bar.Set(resp.Completed)
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set(resp.Completed)
		} else {
			currentDigest = ""
			fmt.Println(resp.Status)
		}
		return nil
	}

	if err := client.Push(context.Background(), &request, fn); err != nil {
		return err
	}
	return nil
}

func ListHandler(cmd *cobra.Command, args []string) error {
	client := api.NewClient()

	models, err := client.List(context.Background())
	if err != nil {
		return err
	}

	var data [][]string

	for _, m := range models.Models {
		if len(args) == 0 || strings.HasPrefix(m.Name, args[0]) {
			data = append(data, []string{m.Name, humanize.Bytes(uint64(m.Size)), format.HumanTime(m.ModifiedAt, "Never")})
		}
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NAME", "SIZE", "MODIFIED"})
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
	client := api.NewClient()

	req := api.DeleteRequest{Name: args[0]}
	if err := client.Delete(context.Background(), &req); err != nil {
		return err
	}
	fmt.Printf("deleted '%s'\n", args[0])
	return nil
}

func CopyHandler(cmd *cobra.Command, args []string) error {
	client := api.NewClient()

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
	client := api.NewClient()

	var currentDigest string
	var bar *progressbar.ProgressBar

	request := api.PullRequest{Name: model, Insecure: insecure}
	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != currentDigest && resp.Digest != "" {
			currentDigest = resp.Digest
			bar = progressbar.DefaultBytes(
				int64(resp.Total),
				fmt.Sprintf("pulling %s...", resp.Digest[7:19]),
			)

			bar.Set(resp.Completed)
		} else if resp.Digest == currentDigest && resp.Digest != "" {
			bar.Set(resp.Completed)
		} else {
			currentDigest = ""
			fmt.Println(resp.Status)
		}
		return nil
	}

	if err := client.Pull(context.Background(), &request, fn); err != nil {
		return err
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
	if len(strings.TrimSpace(prompt)) > 0 {
		client := api.NewClient()

		spinner := NewSpinner("")
		go spinner.Spin(60 * time.Millisecond)

		var latest api.GenerateResponse

		generateContext, ok := cmd.Context().Value(generateContextKey("context")).([]int)
		if !ok {
			generateContext = []int{}
		}

		request := api.GenerateRequest{Model: model, Prompt: prompt, Context: generateContext}
		fn := func(response api.GenerateResponse) error {
			if !spinner.IsFinished() {
				spinner.Finish()
			}

			latest = response

			fmt.Print(response.Response)
			return nil
		}

		if err := client.Generate(context.Background(), &request, fn); err != nil {
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
			}
			return err
		}

		fmt.Println()
		fmt.Println()

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
	}

	return nil
}

func showLayer(l *server.Layer) {
	filename, err := server.GetBlobsPath(l.Digest)
	bts, err := os.ReadFile(filename)
	if err != nil {
		fmt.Printf("Couldn't read layer")
		return
	}
	fmt.Printf(string(bts) + "\n")
}

func generateInteractive(cmd *cobra.Command, model string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	completer := readline.NewPrefixCompleter(
		readline.PcItem("/help"),
		readline.PcItem("/list"),
		readline.PcItem("/set",
			readline.PcItem("history"),
			readline.PcItem("nohistory"),
			readline.PcItem("verbose"),
			readline.PcItem("quiet"),
			readline.PcItem("mode",
				readline.PcItem("vim"),
				readline.PcItem("emacs"),
				readline.PcItem("default"),
			),
		),
		readline.PcItem("/show",
			readline.PcItem("license"),
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

	config := readline.Config{
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
				return nil
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
			multiLineBuffer = strings.TrimPrefix(line, `"""`) + " "
			scanner.SetPrompt("... ")
			continue
		case strings.HasPrefix(line, "/list"):
			args := strings.Fields(line)
			if err := ListHandler(cmd, args[1:]); err != nil {
				return err
			}

			continue
		case strings.HasPrefix(line, "/set"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "history":
					scanner.HistoryEnable()
					continue
				case "nohistory":
					scanner.HistoryDisable()
					continue
				case "verbose":
					cmd.Flags().Set("verbose", "true")
					continue
				case "quiet":
					cmd.Flags().Set("verbose", "false")
					continue
				case "mode":
					if len(args) > 2 {
						switch args[2] {
						case "vim":
							scanner.SetVimMode(true)
							continue
						case "emacs", "default":
							scanner.SetVimMode(false)
							continue
						default:
							usage()
							continue
						}
					} else {
						usage()
						continue
					}
				}
			} else {
				usage()
				continue
			}
		case strings.HasPrefix(line, "/show"):
			args := strings.Fields(line)
			if len(args) > 1 {
				mp := server.ParseModelPath(model)
				manifest, err := server.GetManifest(mp)
				if err != nil {
					fmt.Printf("error: couldn't get a manifestfor this model")
					continue
				}
				switch args[1] {
				case "license":
					for _, l := range manifest.Layers {
						if l.MediaType == "application/vnd.ollama.image.license" {
							showLayer(l)
						}
					}
					continue
				case "system":
					for _, l := range manifest.Layers {
						if l.MediaType == "application/vnd.ollama.image.system" {
							showLayer(l)
						}
					}
					continue
				case "template":
					for _, l := range manifest.Layers {
						if l.MediaType == "application/vnd.ollama.image.template" {
							showLayer(l)
						}
					}
					continue
				default:
					usage()
					continue
				}
			} else {
				usage()
				continue
			}
		case line == "/help", line == "/?":
			usage()
			continue
		case line == "/exit", line == "/bye":
			return nil
		}

		if err := generate(cmd, model, line); err != nil {
			return err
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

// getRunServerParams takes a command and the environment variables and returns the correct params
// given the order of precedence: command line args (highest), environment variables, defaults (lowest)
func getRunServerParams(cmd *cobra.Command) (host, port string, extraOrigins []string, err error) {
	host = os.Getenv("OLLAMA_HOST")
	hostFlag := cmd.Flags().Lookup("host")
	if hostFlag == nil {
		return "", "", nil, errors.New("host unset")
	}
	if hostFlag.Changed || host == "" {
		host = hostFlag.Value.String()
	}
	port = os.Getenv("OLLAMA_PORT")
	portFlag := cmd.Flags().Lookup("port")
	if portFlag == nil {
		return "", "", nil, errors.New("port unset")
	}
	if portFlag.Changed || port == "" {
		port = portFlag.Value.String()
	}
	extraOrigins, err = cmd.Flags().GetStringSlice("origins")
	if err != nil {
		return "", "", nil, err
	}
	return host, port, extraOrigins, nil
}

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port, origins, err := getRunServerParams(cmd)
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%s", host, port))
	if err != nil {
		return err
	}

	return server.Serve(ln, origins)
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
	client := api.NewClient()
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
		Use:          "ollama",
		Short:        "Large language model runner",
		SilenceUsage: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
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

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE:    RunServer,
	}

	serveCmd.Flags().String("port", "11434", "Port to listen on, may also use OLLAMA_PORT environment variable")
	serveCmd.Flags().String("host", "127.0.0.1", "Host listen address, may also use OLLAMA_HOST environment variable")
	serveCmd.Flags().StringSlice("origins", nil, "Additional allowed CORS origins as comma-separated list")

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
		runCmd,
		pullCmd,
		pushCmd,
		listCmd,
		copyCmd,
		deleteCmd,
	)

	return rootCmd
}

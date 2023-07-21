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
	"path/filepath"
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

	request := api.CreateRequest{Name: args[0], Path: filename}
	fn := func(resp api.CreateProgress) error {
		if spinner != nil {
			spinner.Stop()
		}

		spinner = NewSpinner(resp.Status)
		go spinner.Spin(100 * time.Millisecond)

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

	request := api.PushRequest{Name: args[0], Insecure: insecure}
	fn := func(resp api.ProgressResponse) error {
		fmt.Println(resp.Status)
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

	request := api.DeleteRequest{Name: args[0]}
	fn := func(resp api.ProgressResponse) error {
		fmt.Println(resp.Status)
		return nil
	}

	if err := client.Delete(context.Background(), &request, fn); err != nil {
		return err
	}
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

var generateContextKey struct{}

func generate(cmd *cobra.Command, model, prompt string) error {
	if len(strings.TrimSpace(prompt)) > 0 {
		client := api.NewClient()

		spinner := NewSpinner("")
		go spinner.Spin(60 * time.Millisecond)

		var latest api.GenerateResponse

		generateContext, ok := cmd.Context().Value(generateContextKey).([]int)
		if !ok {
			generateContext = []int{}
		}

		request := api.GenerateRequest{Model: model, Prompt: prompt, Context: generateContext}
		fn := func(resp api.GenerateResponse) error {
			if !spinner.IsFinished() {
				spinner.Finish()
			}

			latest = resp

			fmt.Print(resp.Response)

			cmd.SetContext(context.WithValue(cmd.Context(), generateContextKey, resp.Context))
			return nil
		}

		if err := client.Generate(context.Background(), &request, fn); err != nil {
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
	}

	return nil
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
						}
					}
				}
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

func RunServer(_ *cobra.Command, _ []string) error {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "127.0.0.1"
	}

	port := os.Getenv("OLLAMA_PORT")
	if port == "" {
		port = "11434"
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%s", host, port))
	if err != nil {
		return err
	}

	return server.Serve(ln)
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
		Use:   "create MODEL",
		Short: "Create a model from a Modelfile",
		Args:  cobra.MinimumNArgs(1),
		RunE:  CreateHandler,
	}

	createCmd.Flags().StringP("file", "f", "Modelfile", "Name of the Modelfile (default \"Modelfile\")")

	runCmd := &cobra.Command{
		Use:   "run MODEL [PROMPT]",
		Short: "Run a model",
		Args:  cobra.MinimumNArgs(1),
		RunE:  RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")

	serveCmd := &cobra.Command{
		Use:     "serve",
		Aliases: []string{"start"},
		Short:   "Start ollama",
		RunE:    RunServer,
	}

	pullCmd := &cobra.Command{
		Use:   "pull MODEL",
		Short: "Pull a model from a registry",
		Args:  cobra.MinimumNArgs(1),
		RunE:  PullHandler,
	}

	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pushCmd := &cobra.Command{
		Use:   "push MODEL",
		Short: "Push a model to a registry",
		Args:  cobra.MinimumNArgs(1),
		RunE:  PushHandler,
	}

	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List models",
		RunE:    ListHandler,
	}

	deleteCmd := &cobra.Command{
		Use:   "rm",
		Short: "Remove a model",
		Args:  cobra.MinimumNArgs(1),
		RunE:  DeleteHandler,
	}

	rootCmd.AddCommand(
		serveCmd,
		createCmd,
		runCmd,
		pullCmd,
		pushCmd,
		listCmd,
		deleteCmd,
	)

	return rootCmd
}

package cmd

import (
	"archive/tar"
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
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/chzyer/readline"
	"github.com/dustin/go-humanize"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/progressbar"
	"github.com/jmorganca/ollama/server"
	"github.com/jmorganca/ollama/version"
)

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
		if spinner.description != "success" {
			return errors.New("unexpected end to create model")
		}
	}

	return nil
}

func RunHandler(cmd *cobra.Command, args []string) error {
	insecure, err := cmd.Flags().GetBool("insecure")
	if err != nil {
		return err
	}

	mp := server.ParseModelPath(args[0])
	if err != nil {
		return err
	}

	if mp.ProtocolScheme == "http" && !insecure {
		return fmt.Errorf("insecure protocol http")
	}

	fp, err := mp.GetManifestPath(false)
	if err != nil {
		return err
	}

	_, err = os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := pull(args[0], insecure); err != nil {
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
	client, err := api.FromEnv()
	if err != nil {
		return err
	}

	req := api.DeleteRequest{Name: args[0]}
	if err := client.Delete(context.Background(), &req); err != nil {
		return err
	}
	fmt.Printf("deleted '%s'\n", args[0])
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
	if len(strings.TrimSpace(prompt)) > 0 {
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

		if !latest.Done {
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
	}

	return nil
}

func showLayer(l *server.Layer) {
	filename, err := server.GetBlobsPath(l.Digest)
	if err != nil {
		fmt.Println("Couldn't get layer's path")
		return
	}
	bts, err := os.ReadFile(filename)
	if err != nil {
		fmt.Println("Couldn't read layer")
		return
	}
	fmt.Println(string(bts))
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
				if err != nil {
					return err
				}

				manifest, err := server.GetManifest(mp)
				if err != nil {
					fmt.Println("error: couldn't get a manifest for this model")
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

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port := "127.0.0.1", "11434"

	parts := strings.Split(os.Getenv("OLLAMA_HOST"), ":")
	if ip := net.ParseIP(parts[0]); ip != nil {
		host = ip.String()
	}

	if len(parts) > 1 {
		port = parts[1]
	}

	// deprecated: include port in OLLAMA_HOST
	if p := os.Getenv("OLLAMA_PORT"); p != "" {
		port = p
	}

	err := initializeKeypair()
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%s", host, port))
	if err != nil {
		return err
	}

	var origins []string
	if o := os.Getenv("OLLAMA_ORIGINS"); o != "" {
		origins = strings.Split(o, ",")
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

		err = os.MkdirAll(path.Dir(privKeyPath), 0o700)
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

func handleFile(tarWriter *tar.Writer, filePath string, tarPath string) error {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer file.Close()

	// Get the file info
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file info for %s: %w", filePath, err)
	}

	// Create a tar header from the file info
	header, err := tar.FileInfoHeader(info, info.Name())
	if err != nil {
		return fmt.Errorf("failed to create tar header for file %s: %w", filePath, err)
	}
	header.Name = tarPath

	// Write the header to the tarball
	err = tarWriter.WriteHeader(header)
	if err != nil {
		return fmt.Errorf("failed to write tar header for file %s: %w", filePath, err)
	}

	// Write the file content to the tarball
	_, err = io.Copy(tarWriter, file)
	if err != nil {
		return fmt.Errorf("failed to write file %s to tarball: %w", filePath, err)
	}

	return nil
}

func ExportHandler(cmd *cobra.Command, args []string) error {
	modelName := args[0]
	filePath := args[1]

	// Get the manifest for the model
	mp := server.ParseModelPath(modelName)
	mf, err := server.GetManifest(mp)
	if err != nil {
		return fmt.Errorf("failed to get manifest for model %s: %w", modelName, err)
	}

	// Derive the manifest path
	manifestPath, err := mp.GetManifestPath(false)
	if err != nil {
		return fmt.Errorf("failed to get manifest path for model %s: %w", modelName, err)
	}

	tarFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create tar file at %s: %w", filePath, err)
	}
	defer tarFile.Close()

	tarWriter := tar.NewWriter(tarFile)
	defer tarWriter.Close()

	// Add the manifest to the tarball
	err = handleFile(tarWriter, manifestPath, "manifest.json")
	if err != nil {
		return err
	}

	// Get the home directory
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	// Add each blob to the tarball
	for _, layer := range mf.Layers {
		blobPath := filepath.Join(home, ".ollama/models/blobs", layer.Digest)
		err = handleFile(tarWriter, blobPath, "blobs/"+layer.Digest)
		if err != nil {
			return err
		}
	}

	return nil
}

func ImportHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 2 {
		return errors.New("incorrect number of arguments. Expected MODELNAME:TAG and FILEPATH")
	}

	modelTag := strings.Split(args[0], ":")
	if len(modelTag) != 2 {
		return errors.New("incorrect format for MODELNAME:TAG. It should be in the format 'model:tag'")
	}

	newModelName := modelTag[0]
	newVariant := modelTag[1]
	filePath := args[1]

	// Open the tar file
	tarFile, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open tar file at %s: %w", filePath, err)
	}
	defer tarFile.Close()

	// Create a tar reader
	tarReader := tar.NewReader(tarFile)

	// Get the home directory
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	// Extract each file from the tarball
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return fmt.Errorf("failed to read next file from tarball: %w", err)
		}

		// Derive the file path based on whether the file is a blob or the manifest
		var filePath string
		if strings.HasPrefix(header.Name, "blobs/") {
			blobName := strings.TrimPrefix(header.Name, "blobs/")
			// This is a blob, place it in the blobs directory
			filePath = filepath.Join(home, ".ollama/models/blobs", blobName)
		} else if header.Name == "manifest.json" {
			// This is the manifest, place it in the manifests directory and rename it to the variant
			filePath = filepath.Join(home, ".ollama/models/manifests/registry.ollama.ai/library", newModelName, newVariant)
		} else {
			// Unexpected file, skip it
			continue
		}

		// Create the directory if it doesn't exist
		dir := filepath.Dir(filePath)
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			err = os.MkdirAll(dir, 0755)
			if err != nil {
				return fmt.Errorf("failed to create directory %s: %w", dir, err)
			}
		}

		// Create the file
		file, err := os.Create(filePath)
		if err != nil {
			return fmt.Errorf("failed to create file %s: %w", filePath, err)
		}
		defer file.Close()

		// Write the file content
		_, err = io.Copy(file, tarReader)
		if err != nil {
			return fmt.Errorf("failed to write file %s: %w", filePath, err)
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

	runCmd := &cobra.Command{
		Use:     "run MODEL [PROMPT]",
		Short:   "Run a model",
		Args:    cobra.MinimumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    RunHandler,
	}

	runCmd.Flags().Bool("verbose", false, "Show timings for response")
	runCmd.Flags().Bool("insecure", false, "Use an insecure registry")

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

	exportCmd := &cobra.Command{
		Use:   "export MODEL:TAG FILEPATH",
		Short: "Export a model to a file.",
		Long: `Use this to transfer a model between Ollama installations. 
The export bundle destination can be any valid file path, but must end with .ollamabundle.

  Example: ollama export llama2:7b /path/to/myExportedLlama-7b.ollamabundle`,
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    ExportHandler,
	}

	importCmd := &cobra.Command{
		Use:     "import MODELNAME:TAG FILEPATH",
		Short:   "Import a model from a file",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    ImportHandler,
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
		importCmd,
		exportCmd,
	)

	return rootCmd
}

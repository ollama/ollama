package cmd

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/mattn/go-runewidth"
	"github.com/olekukonko/tablewriter"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

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

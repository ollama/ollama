package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
)

const askMaxBytes = 64 << 10

func newAskCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "ask MODEL QUESTION [TEXT]",
		Short: "Ask a decision model about text",
		Long: `Ask a decision model a yes/no question, choose a label, or score text against a rubric.
Quote QUESTION and TEXT separately. You can also supply text through stdin.
If you supply QUESTION without TEXT or stdin, the question is also the input.

To ask multiple questions about one input, use:
  ollama ask MODEL --questions FILE [TEXT]
FILE contains the System One questions object with type, instructions, and criteria.
Use only one of --choice, --score, or --questions.

In a terminal, answers include probabilities or the score scale.
If stdout is redirected, output contains only the yes probability (0–1), label, or score.
Use --json to print the full API response.
With --questions, redirected output uses JSON by default.

Successful answers return exit status 0, regardless of the probability or score.
Errors appear on stderr. The command returns a nonzero exit status for errors.
If the model is unavailable, use ollama pull MODEL.`,
		Example: `  ollama ask nimble "Is water made of hydrogen and oxygen?"
  ollama ask nimble "Is this a refund request?" "I was charged twice."
  cat ticket.txt | ollama ask nimble "Is this urgent?"
  ollama ask nimble "Which label fits?" "Checkout returns 500 errors." --choice bug --choice billing --choice account
  ollama ask nimble "How urgent is this?" "Checkout is down." --score Routine --score Soon --score Immediate
  ollama ask nimble --questions questions.json "I was charged twice." --json`,
		RunE: askHandler,
	}
	cmd.Flags().StringArray("choice", nil, "Label or label=description (repeat for each choice)")
	cmd.Flags().StringArray("score", nil, "Score criterion (repeat in order from lowest to highest)")
	cmd.Flags().String("questions", "", "Read the System One questions object from a JSON file")
	cmd.Flags().Bool("json", false, "Print the full JSON response with probabilities and token usage")
	cmd.MarkFlagsMutuallyExclusive("choice", "score", "questions")
	return cmd
}

func askHandler(cmd *cobra.Command, args []string) error {
	req, err := readAskRequest(cmd, args)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()
	cmd.SetContext(ctx)
	if err := checkServerHeartbeat(cmd, nil); err != nil {
		return err
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	info, err := client.Show(ctx, &api.ShowRequest{Model: req.Model})
	if err != nil {
		var status api.StatusError
		if errors.As(err, &status) && status.StatusCode == http.StatusNotFound {
			return fmt.Errorf("model %q not found; pull it with 'ollama pull %s'", req.Model, req.Model)
		}
		return err
	}
	if !slices.Contains(info.Capabilities, model.CapabilityDecision) {
		return fmt.Errorf("model %q does not support decisions; choose a decision model (e.g. ollama ask nimble \"Is water made of hydrogen and oxygen?\")", req.Model)
	}

	jsonOutput, _ := cmd.Flags().GetBool("json")
	human := askTerminal(cmd.OutOrStdout()) && !jsonOutput
	var p *progress.Progress
	if human && askTerminal(cmd.ErrOrStderr()) {
		p = progress.NewProgress(cmd.ErrOrStderr())
		p.Add("", progress.NewSpinner("Asking "+req.Model))
	}
	resp, err := client.SystemOne(ctx, req)
	if p != nil {
		p.StopAndClear()
	}
	if err != nil {
		return fmt.Errorf("ask %s: %w", req.Model, err)
	}
	return writeAskResponse(cmd.OutOrStdout(), req, resp, human, jsonOutput || (!human && cmd.Flags().Changed("questions")))
}

func askTerminal(stream any) bool {
	f, ok := stream.(*os.File)
	return ok && term.IsTerminal(int(f.Fd()))
}

func readAskRequest(cmd *cobra.Command, args []string) (*api.SystemOneRequest, error) {
	if len(args) == 0 {
		return nil, errors.New("use: ollama ask MODEL QUESTION [TEXT]; quote the question and text, or pipe text through stdin")
	}
	req := &api.SystemOneRequest{Model: args[0], Questions: &api.SystemOneQuestions{}}
	args = args[1:]
	var textArgs []string
	var questionText string
	if cmd.Flags().Changed("questions") {
		if len(args) > 1 {
			return nil, errors.New("use: ollama ask MODEL --questions FILE [TEXT]; quote text containing spaces")
		}
		filename, _ := cmd.Flags().GetString("questions")
		f, err := os.Open(filename)
		if err != nil {
			return nil, fmt.Errorf("read questions: %w", err)
		}
		data, err := io.ReadAll(io.LimitReader(f, askMaxBytes+1))
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("read questions: %w", err)
		}
		if len(data) > askMaxBytes {
			return nil, errors.New("questions file must fit within the 64 KiB request limit")
		}
		var questions orderedmap.Map[string, json.RawMessage]
		if err := json.Unmarshal(data, &questions); err != nil {
			return nil, fmt.Errorf("read questions: %w", err)
		}
		for name, raw := range questions.All() {
			var question api.SystemOneQuestion
			decoder := json.NewDecoder(bytes.NewReader(raw))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&question); err != nil {
				return nil, fmt.Errorf("read question %q: %w", name, err)
			}
			req.Questions.Set(name, question)
		}
		textArgs = args
	} else {
		if len(args) < 1 || len(args) > 2 {
			return nil, errors.New("use: ollama ask MODEL QUESTION [TEXT]; quote the question and text, or pipe text through stdin")
		}
		questionText = args[0]
		question := api.SystemOneQuestion{Type: "noul"}
		question.Instructions, _ = json.Marshal(args[0])
		choices, _ := cmd.Flags().GetStringArray("choice")
		scores, _ := cmd.Flags().GetStringArray("score")
		switch {
		case cmd.Flags().Changed("choice"):
			question.Type = "choice"
			criteria := &orderedmap.Map[string, string]{}
			for _, choice := range choices {
				label, description, hasDescription := strings.Cut(choice, "=")
				if _, exists := criteria.Get(label); exists {
					return nil, fmt.Errorf("duplicate choice %q", label)
				}
				if !hasDescription {
					description = label
				}
				criteria.Set(label, description)
			}
			question.Criteria, _ = json.Marshal(criteria)
		case cmd.Flags().Changed("score"):
			question.Type = "score"
			question.Criteria, _ = json.Marshal(scores)
		}
		req.Questions.Set("answer", question)
		textArgs = args[1:]
	}

	var text string
	if len(textArgs) > 0 {
		text = textArgs[0]
	} else if askTerminal(cmd.InOrStdin()) {
		text = questionText
	} else {
		data, err := io.ReadAll(io.LimitReader(cmd.InOrStdin(), askMaxBytes+1))
		if err != nil {
			return nil, fmt.Errorf("read text from stdin: %w", err)
		}
		text = string(data)
		if len(data) == 0 {
			text = questionText
		}
	}
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("text must not be empty; provide TEXT or pipe text through stdin")
	}
	if len(text) > askMaxBytes {
		return nil, errors.New("text must fit within the 64 KiB request limit")
	}
	req.State, _ = json.Marshal(text)
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	if len(data) > askMaxBytes {
		return nil, errors.New("request must fit within 64 KiB; shorten the text or questions")
	}
	if _, err := decision.Compile(*req); err != nil {
		return nil, err
	}
	return req, nil
}

func writeAskResponse(w io.Writer, req *api.SystemOneRequest, resp *api.SystemOneResponse, human, jsonOutput bool) error {
	// Validate every answer before writing, so a failed request never leaves
	// a partial set of values on stdout.
	var output strings.Builder
	for name, question := range req.Questions.All() {
		raw, ok := resp.Answers.Get(name)
		if !ok {
			return fmt.Errorf("server returned no answer for %q", name)
		}
		var answer struct {
			Type          string             `json:"type"`
			Noul          *float64           `json:"noul"`
			Choice        *string            `json:"choice"`
			Score         *float64           `json:"score"`
			Probabilities map[string]float64 `json:"probabilities"`
		}
		if err := json.Unmarshal(raw, &answer); err != nil {
			return fmt.Errorf("decode answer %q: %w", name, err)
		}
		if answer.Type != question.Type {
			return fmt.Errorf("server returned an unexpected answer type for %q", name)
		}
		var value string
		switch answer.Type {
		case "noul":
			if answer.Noul == nil || *answer.Noul < 0 || *answer.Noul > 1 {
				return fmt.Errorf("server returned an invalid yes probability for %q", name)
			}
			value = fmt.Sprintf("%g", *answer.Noul)
			if human {
				value = fmt.Sprintf("Yes: %.2f%%", *answer.Noul*100)
			}
		case "choice":
			if answer.Choice == nil {
				return fmt.Errorf("server returned no choice for %q", name)
			}
			var criteria map[string]json.RawMessage
			json.Unmarshal(question.Criteria, &criteria)
			if _, ok := criteria[*answer.Choice]; !ok {
				return fmt.Errorf("server returned an unknown choice for %q", name)
			}
			value = *answer.Choice
			if human {
				if probability, ok := answer.Probabilities[value]; ok {
					value = fmt.Sprintf("%s (%.2f%% probability)", value, probability*100)
				}
			}
		case "score":
			var criteria []string
			json.Unmarshal(question.Criteria, &criteria)
			maximum := len(criteria) - 1
			if answer.Score == nil || *answer.Score < 0 || *answer.Score > float64(maximum) {
				return fmt.Errorf("server returned an invalid score for %q", name)
			}
			value = fmt.Sprintf("%g", *answer.Score)
			if human {
				value = fmt.Sprintf("%.2f / %d", *answer.Score, maximum)
			}
		}
		if human && req.Questions.Len() > 1 {
			fmt.Fprintf(&output, "%s: ", name)
		}
		fmt.Fprintln(&output, value)
	}
	if jsonOutput {
		encoder := json.NewEncoder(w)
		encoder.SetEscapeHTML(false)
		return encoder.Encode(resp)
	}
	_, err := io.WriteString(w, output.String())
	return err
}

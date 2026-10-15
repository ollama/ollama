package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/progress"
)

// runDecision runs a decision model, which answers the --ask questions about
// a state: the text after the model name, or stdin.
func runDecision(cmd *cobra.Command, opts runOptions, interactive bool) error {
	opts.Questions = askedQuestions(cmd)
	if interactive {
		opts.Decision = true
		return generateInteractive(cmd, opts)
	}
	if opts.Prompt == "" || opts.Questions.Len() == 0 {
		return fmt.Errorf("%s answers questions about text, e.g. ollama run %s --ask \"Is a refund being requested?\" < ticket.txt", opts.Model, opts.Model)
	}

	input := api.Message{Role: "user", Content: opts.Prompt}
	if opts.MultiModal {
		var err error
		if input.Content, input.Images, err = extractFileData(input.Content); err != nil {
			return err
		}
	}
	return decide(cmd, opts, input, opts.Questions)
}

// yesNoQuestions asks each text as a yes/no question keyed by the text itself.
func yesNoQuestions(texts ...string) *api.SystemOneQuestions {
	questions := &api.SystemOneQuestions{}
	for _, text := range texts {
		instructions, _ := json.Marshal(text)
		questions.Set(text, api.SystemOneQuestion{Type: "noul", Instructions: instructions})
	}
	return questions
}

// questionFlag is --ask, --choice or --level. The three share the questions
// being asked, so each --choice or --level answers the --ask before it.
type questionFlag struct {
	name  string
	asked *[]askedQuestion
}

type askedQuestion struct {
	text    string
	choices *orderedmap.Map[string, any]
	levels  []string
}

func (f questionFlag) String() string { return "" }
func (f questionFlag) Type() string   { return "string" }

func (f questionFlag) Set(value string) error {
	if f.name == "ask" {
		*f.asked = append(*f.asked, askedQuestion{text: value})
		return nil
	}
	if len(*f.asked) == 0 {
		return errors.New("must come after the --ask it answers")
	}
	q := &(*f.asked)[len(*f.asked)-1]
	if f.name == "level" {
		q.levels = append(q.levels, value)
	} else {
		if q.choices == nil {
			q.choices = orderedmap.New[string, any]()
		}
		// A label may have a description: billing="Payments and refunds"
		if label, description, ok := strings.Cut(value, "="); ok {
			q.choices.Set(label, description)
		} else {
			q.choices.Set(value, nil)
		}
	}
	if q.choices != nil && q.levels != nil {
		return errors.New("a question takes --choice or --level, not both")
	}
	return nil
}

// askedQuestions returns the --ask questions keyed by their text: yes/no, or
// answered with a --choice label or on a --level scale.
func askedQuestions(cmd *cobra.Command) *api.SystemOneQuestions {
	questions := &api.SystemOneQuestions{}
	flag := cmd.Flags().Lookup("ask")
	if flag == nil {
		return questions
	}
	for _, q := range *flag.Value.(questionFlag).asked {
		question := api.SystemOneQuestion{Type: "noul"}
		question.Instructions, _ = json.Marshal(q.text)
		switch {
		case q.choices != nil:
			question.Type = "choice"
			question.Criteria, _ = json.Marshal(q.choices)
		case q.levels != nil:
			question.Type = "score"
			question.Criteria, _ = json.Marshal(q.levels)
		}
		questions.Set(q.text, question)
	}
	return questions
}

// decide asks a decision model questions about a state and prints each answer:
// the likelier answer to a yes/no question, the chosen label, or the most
// likely score level, with its probability, and for a score the score itself.
func decide(cmd *cobra.Command, opts runOptions, input api.Message, questions *api.SystemOneQuestions) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	// Piped text and multiline messages end in a newline that isn't part of the state.
	text := strings.TrimRight(input.Content, "\r\n")
	state, _ := json.Marshal(text)
	if s := strings.TrimSpace(text); strings.HasPrefix(s, "{") || strings.HasPrefix(s, "[") {
		if json.Valid([]byte(s)) {
			state = json.RawMessage(s)
		}
	}
	req := &api.SystemOneRequest{
		Model:     opts.Model,
		Runner:    opts.Runner,
		State:     state,
		Images:    input.Images,
		Questions: questions,
		KeepAlive: opts.KeepAlive,
	}

	ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
	defer stop()

	p := progress.NewProgress(os.Stderr)
	p.Add("", progress.NewSpinner(""))
	start := time.Now()
	resp, err := client.SystemOne(ctx, req)
	p.StopAndClear()
	if errors.Is(err, context.Canceled) {
		return nil
	} else if err != nil {
		return err
	}

	if opts.Format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetEscapeHTML(false)
		return enc.Encode(resp)
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	for name := range questions.All() {
		raw, ok := resp.Answers.Get(name)
		if !ok {
			return fmt.Errorf("no answer to %q", name)
		}
		var a struct {
			Type          string             `json:"type"`
			Noul          float64            `json:"noul"`
			Choice        string             `json:"choice"`
			Score         float64            `json:"score"`
			Legend        map[string]any     `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return fmt.Errorf("answer to %q: %w", name, err)
		}
		var answer, measure string
		switch a.Type {
		case "noul":
			answer, measure = "yes", fmt.Sprintf("%.0f%%", a.Noul*100)
			if a.Noul < 0.5 {
				answer, measure = "no", fmt.Sprintf("%.0f%%", (1-a.Noul)*100)
			}
		case "choice":
			answer, measure = a.Choice, fmt.Sprintf("%.0f%%", a.Probabilities[a.Choice]*100)
		case "score":
			// Not the level nearest the score: an answer split between the
			// ends of the scale scores in the middle.
			level := "0"
			for i := 1; i < len(a.Legend); i++ {
				if k := strconv.Itoa(i); a.Probabilities[k] > a.Probabilities[level] {
					level = k
				}
			}
			if v, ok := a.Legend[level]; ok {
				answer = fmt.Sprint(v)
			}
			measure = fmt.Sprintf("%.0f%%\t%.1f / %d", a.Probabilities[level]*100, a.Score, len(a.Legend)-1)
		default:
			return fmt.Errorf("answer to %q has unknown type %q", name, a.Type)
		}
		if questions.Len() > 1 {
			fmt.Fprintf(tw, "%s\t", name)
		}
		fmt.Fprintf(tw, "%s\t%s\n", answer, measure)
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	if verbose, _ := cmd.Flags().GetBool("verbose"); verbose {
		metrics := api.Metrics{
			TotalDuration:         time.Since(start),
			PromptEvalCount:       resp.Usage.InputTokens,
			PromptEvalCachedCount: resp.PromptEvalCachedCount,
		}
		metrics.Summary()
	}
	return nil
}

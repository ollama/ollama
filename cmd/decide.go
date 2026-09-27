package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/progress"
)

// decide asks a decision model yes-or-no questions about text with the System
// One API and prints each answer on its own line.
func decide(ctx context.Context, opts runOptions, text string, questions ...string) error {
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()

	fields := orderedmap.New[string, map[string]string]()
	for _, q := range questions {
		fields.Set(q, map[string]string{"type": "noul", "instructions": q})
	}
	body, err := json.Marshal(map[string]any{
		"model":      opts.Model,
		"state":      text,
		"questions":  fields,
		"keep_alive": opts.KeepAlive,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, envconfig.Host().JoinPath("/v1/systemone").String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	p := progress.NewProgress(os.Stderr)
	p.Add("", progress.NewSpinner(""))
	resp, err := http.DefaultClient.Do(req)
	p.StopAndClear()
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(data, &e) != nil || e.Error == "" {
			e.Error = strings.TrimSpace(string(data))
		}
		return api.StatusError{StatusCode: resp.StatusCode, Status: resp.Status, ErrorMessage: e.Error}
	}

	if opts.Format == "json" {
		fmt.Println(string(data))
		return nil
	}
	var result struct {
		Answers map[string]struct {
			Noul float64 `json:"noul"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return err
	}
	for _, q := range questions {
		answer, ok := result.Answers[q]
		switch {
		case !ok:
			return fmt.Errorf("no answer to %q", q)
		case answer.Noul >= 0.5:
			fmt.Printf("yes (%.0f%%)\n", answer.Noul*100)
		default:
			fmt.Printf("no (%.0f%%)\n", (1-answer.Noul)*100)
		}
	}
	return nil
}

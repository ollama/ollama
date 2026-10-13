package decisiontest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
)

type Response struct {
	Model                 string            `json:"model"`
	Answers               map[string]Answer `json:"answers"`
	Usage                 decision.Usage    `json:"usage"`
	PromptEvalCachedCount *int              `json:"prompt_eval_cached_count"`
}

// Pointers distinguish missing/null numbers from valid zero values.
type Answer struct {
	Type          string              `json:"type"`
	Choice        string              `json:"choice"`
	Noul          *float64            `json:"noul"`
	Score         *float64            `json:"score"`
	Confidence    *float64            `json:"confidence"`
	Probabilities map[string]*float64 `json:"probabilities"`
	Legend        map[string]any      `json:"legend"`
}

// Do measures the complete HTTP round trip, excluding request encoding and
// response validation. The elapsed time includes network and server queueing.
func Do(ctx context.Context, endpoint string, input decision.Request) (Response, time.Duration, error) {
	var result Response
	body, err := json.Marshal(input)
	if err != nil {
		return result, 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return result, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return result, time.Since(start), err
	}
	defer resp.Body.Close()
	const limit = 8 << 20
	body, err = io.ReadAll(io.LimitReader(resp.Body, limit+1))
	elapsed := time.Since(start)
	if err != nil {
		return result, elapsed, err
	}
	if len(body) > limit {
		return result, elapsed, fmt.Errorf("decision response exceeds %d bytes", limit)
	}
	if resp.StatusCode != http.StatusOK {
		return result, elapsed, api.StatusError{StatusCode: resp.StatusCode, Status: resp.Status, ErrorMessage: string(body)}
	}
	result, err = decode(input, body)
	return result, elapsed, err
}

func decode(input decision.Request, body []byte) (Response, error) {
	result := Response{Usage: decision.Usage{InputTokens: -1, OutputTokens: -1}}
	if err := json.Unmarshal(body, &result); err != nil {
		return result, err
	}
	if result.Model != input.Model || len(result.Answers) != input.Questions.Len() || result.Usage.InputTokens <= 0 || result.Usage.OutputTokens < 0 {
		return result, fmt.Errorf("invalid response model, answer count, or token usage")
	}
	if n := result.PromptEvalCachedCount; n != nil && (*n < 0 || *n > result.Usage.InputTokens) {
		return result, fmt.Errorf("cached token count outside input token range")
	}
	for name, q := range input.Questions.All() {
		if err := result.Answers[name].validate(q); err != nil {
			return result, fmt.Errorf("answer %q: %w", name, err)
		}
	}
	return result, nil
}

func probability(p *float64) bool {
	return p != nil && *p >= 0 && *p <= 1
}

func (a Answer) validate(q decision.Question) error {
	if a.Type != q.Type {
		return fmt.Errorf("type %q, want %q", a.Type, q.Type)
	}
	if q.Type == "noul" {
		if !probability(a.Noul) {
			return fmt.Errorf("missing or invalid noul probability")
		}
		return nil
	}
	keys := criteria(q)
	if len(a.Probabilities) != len(keys) || !probability(a.Confidence) {
		return fmt.Errorf("missing or invalid probabilities or confidence")
	}
	var sum, weighted, best float64
	for key := range keys {
		p := a.Probabilities[key]
		if !probability(p) {
			return fmt.Errorf("missing or invalid probability for %q", key)
		}
		sum += *p
		best = max(best, *p)
		if q.Type == "score" {
			i, _ := strconv.Atoi(key)
			weighted += float64(i) * *p
		}
	}
	// Some implementations report four decimal places. Account for rounding
	// each probability independently; confidence is model-defined.
	const rounding = 0.5e-4
	if math.Abs(sum-1) > 1e-6+float64(len(keys))*rounding {
		return fmt.Errorf("probabilities sum to %g, want 1", sum)
	}
	if q.Type == "choice" {
		if p := a.Probabilities[a.Choice]; p == nil || *p < best {
			return fmt.Errorf("choice %q is not a probability maximum", a.Choice)
		}
	} else {
		// Include rounding of the score and its weighted probabilities.
		n := len(keys)
		tolerance := 1e-6 + rounding*float64(1+n*(n-1)/2)
		if a.Score == nil || *a.Score < 0 || *a.Score > float64(n-1) || math.Abs(*a.Score-weighted) > tolerance {
			return fmt.Errorf("score does not match probabilities")
		}
		if !reflect.DeepEqual(a.Legend, keys) {
			return fmt.Errorf("legend does not match criteria")
		}
	}
	return nil
}

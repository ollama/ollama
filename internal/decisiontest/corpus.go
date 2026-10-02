// Package decisiontest reads and scores external System One evaluation corpora.
package decisiontest

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/decision"
	_ "golang.org/x/image/webp"
)

type Case struct {
	ID          string              `json:"id"`
	Dataset     string              `json:"dataset"`
	Request     decision.Request    `json:"request"`
	ImageFiles  []string            `json:"image_files,omitempty"`
	Expected    map[string]Expected `json:"expected"`
	ImageBytes  int                 `json:"-"`
	ImagePixels int64               `json:"-"`
}

type Expected struct {
	Choice *string  `json:"choice,omitempty"`
	Noul   *bool    `json:"noul,omitempty"`
	Min    *float64 `json:"min,omitempty"`
	Max    *float64 `json:"max,omitempty"`
	Level  *int     `json:"level,omitempty"`
}

type Corpus struct {
	Cases  []Case
	SHA256 string
}

// Load resolves image_files relative to the JSONL file, before any timed requests.
// The digest covers both the JSONL bytes and the referenced image contents.
func Load(path string) (Corpus, error) {
	var corpus Corpus
	f, err := os.Open(path)
	if err != nil {
		return corpus, err
	}
	defer f.Close()
	hash := sha256.New()
	scanner := bufio.NewScanner(io.TeeReader(f, hash))
	scanner.Buffer(make([]byte, 4096), 16<<20)
	ids := make(map[string]bool)
	for line := 1; scanner.Scan(); line++ {
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			continue
		}
		var c Case
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&c); err != nil {
			return corpus, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			return corpus, fmt.Errorf("%s:%d: expected one JSON object", path, line)
		}
		if c.ID == "" || c.Dataset == "" || ids[c.ID] {
			return corpus, fmt.Errorf("%s:%d: require dataset and unique nonempty id", path, line)
		}
		ids[c.ID] = true
		if err := c.validate(); err != nil {
			return corpus, fmt.Errorf("%s:%d (%s): %w", path, line, c.ID, err)
		}
		corpus.Cases = append(corpus.Cases, c)
	}
	if err := scanner.Err(); err != nil {
		return corpus, fmt.Errorf("%s: %w", path, err)
	}
	if len(corpus.Cases) == 0 {
		return corpus, fmt.Errorf("%s: empty decision corpus", path)
	}
	for i := range corpus.Cases {
		c := &corpus.Cases[i]
		for _, name := range c.ImageFiles {
			if !filepath.IsLocal(name) {
				return corpus, fmt.Errorf("%s: image path must be relative to the corpus: %q", c.ID, name)
			}
			data, err := os.ReadFile(filepath.Join(filepath.Dir(path), name))
			if err != nil {
				return corpus, fmt.Errorf("%s: %w", c.ID, err)
			}
			c.Request.Images = append(c.Request.Images, data)
			// Frame each image so different file boundaries cannot hash alike.
			_, _ = fmt.Fprintf(hash, "\x00%d:", len(data))
			_, _ = hash.Write(data)
		}
		for _, data := range c.Request.Images {
			config, _, err := image.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return corpus, fmt.Errorf("%s: image: %w", c.ID, err)
			}
			c.ImageBytes += len(data)
			c.ImagePixels += int64(config.Width) * int64(config.Height)
		}
	}
	corpus.SHA256 = hex.EncodeToString(hash.Sum(nil))
	return corpus, nil
}

func (c Case) validate() error {
	req := c.Request
	if req.Model != "" || req.KeepAlive != nil {
		return fmt.Errorf("model and keep_alive belong to the test invocation, not the corpus")
	}
	req.Model = "validation"
	// Clef accepts the full decision schema, including optional instructions and
	// images. Model-specific limits remain the server's responsibility.
	if _, err := decision.CompileWithEncoder(req, "clef"); err != nil {
		return err
	}
	if len(c.Expected) != req.Questions.Len() {
		return fmt.Errorf("require an expectation for every question")
	}
	for name, q := range req.Questions.All() {
		want, ok := c.Expected[name]
		if !ok {
			return fmt.Errorf("missing expectation for %q", name)
		}
		keys := criteria(q)
		valid := false
		switch q.Type {
		case "choice":
			if want.Choice != nil {
				_, valid = keys[*want.Choice]
			}
			valid = valid && want.Noul == nil && want.Min == nil && want.Max == nil && want.Level == nil
		case "noul":
			valid = want.Noul != nil && want.Choice == nil && want.Min == nil && want.Max == nil && want.Level == nil
		case "score":
			valid = want.Min != nil && want.Max != nil && want.Level == nil && *want.Min >= 0 && *want.Min <= *want.Max && *want.Max <= float64(len(keys)-1)
			if want.Level != nil {
				valid = want.Min == nil && want.Max == nil && *want.Level >= 0 && *want.Level < len(keys)
			}
			valid = valid && want.Choice == nil && want.Noul == nil
		}
		if !valid {
			return fmt.Errorf("invalid %s expectation for %q", q.Type, name)
		}
	}
	return nil
}

// criteria is only called after validating the request schema.
func criteria(q decision.Question) map[string]any {
	keys := make(map[string]any)
	switch q.Type {
	case "choice":
		_ = json.Unmarshal(q.Criteria, &keys)
	case "score":
		var labels []any
		_ = json.Unmarshal(q.Criteria, &labels)
		for i, label := range labels {
			keys[strconv.Itoa(i)] = label
		}
	case "noul":
		keys["false"], keys["true"] = "No", "Yes"
	}
	return keys
}

func (c Case) Options() int {
	var n int
	for _, q := range c.Request.Questions.All() {
		n += len(criteria(q))
	}
	return n
}

// Mismatches compares labels, never the model's confidence in those labels.
// result must have passed response validation in Do.
func (c Case) Mismatches(result Response) []string {
	var wrong []string
	for name, want := range c.Expected {
		a := result.Answers[name]
		switch {
		case want.Choice != nil && a.Choice != *want.Choice:
			wrong = append(wrong, fmt.Sprintf("%s: choice %q, want %q", name, a.Choice, *want.Choice))
		case want.Noul != nil && (*a.Noul == .5 || (*a.Noul > .5) != *want.Noul):
			wrong = append(wrong, fmt.Sprintf("%s: noul %g, want %v", name, *a.Noul, *want.Noul))
		case want.Min != nil && (*a.Score < *want.Min || *a.Score > *want.Max):
			wrong = append(wrong, fmt.Sprintf("%s: score %g, want [%g,%g]", name, *a.Score, *want.Min, *want.Max))
		case want.Level != nil:
			key := strconv.Itoa(*want.Level)
			for other, p := range a.Probabilities {
				if other != key && *p >= *a.Probabilities[key] {
					wrong = append(wrong, fmt.Sprintf("%s: most probable level is not uniquely %d", name, *want.Level))
					break
				}
			}
		}
	}
	return wrong
}

func (c Corpus) HasImages() bool {
	for _, row := range c.Cases {
		if len(row.Request.Images) > 0 {
			return true
		}
	}
	return false
}

// Mode validates the benchmark's label policy.
func Mode(value string) (string, error) {
	if value == "" {
		value = "regression"
	}
	if value != "regression" && value != "score" {
		return "", fmt.Errorf("decision mode must be regression or score, got %q", strings.TrimSpace(value))
	}
	return value, nil
}

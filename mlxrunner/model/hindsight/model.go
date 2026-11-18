// Package hindsight runs Hindsight's decision checkpoints (hs-decide-1-flash):
// a causal Qwen3.5 decoder that reads a request's state and every question in
// one sequence and answers each question at the colon of its "Qn:" line,
// through the tied LM head's rows for the option codes.
package hindsight

import (
	"fmt"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

const formatID = "hindsight-decision-v1"

type config struct {
	FormatVersion    int    `json:"format_version"`
	FormatID         string `json:"format_id"`
	PromptVersion    int    `json:"prompt_version"`
	Codes            string `json:"codes"`
	QuestionLabels   string `json:"question_labels"`
	PlainText        bool   `json:"plain_text"`
	AttentionMode    string `json:"attention_mode"`
	MaxRequestTokens int    `json:"max_request_tokens"`
}

type Model struct {
	*qwen3_5.Model
	config config
	tok    *tokenizer.Tokenizer
	codes  []string         // choice codes, A to Z then two-letter codes
	single map[string]int32 // code spellings to their single token, -1 when several
}

func NewModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("decision_config.json", &cfg); err != nil {
		return nil, fmt.Errorf("read Hindsight decision config: %w", err)
	}
	if cfg.FormatVersion != 1 || cfg.FormatID != formatID || cfg.PromptVersion != 2 || cfg.Codes != "pairs" ||
		cfg.QuestionLabels != "positions" || !cfg.PlainText || cfg.AttentionMode != "causal" || cfg.MaxRequestTokens < 1 {
		return nil, fmt.Errorf("unsupported Hindsight decision config (format %q, prompt version %d)", cfg.FormatID, cfg.PromptVersion)
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	if m.Config.ModelType != "qwen3_5_text" {
		return nil, fmt.Errorf("unsupported Hindsight decision backbone %q", m.Config.ModelType)
	}
	// Without it the linear-attention decay drifts on long requests: one answer
	// of a 21,724-token request read 0.10 here against 0.71 in PyTorch.
	m.Config.PreciseGatedDeltaDecay = true
	if err := m.initCodes(m.Tokenizer()); err != nil {
		return nil, err
	}
	return m, nil
}

// initCodes builds the choice code table and every code spelling's token, so
// requests only read them.
func (m *Model) initCodes(tok *tokenizer.Tokenizer) error {
	m.tok = tok
	m.single = map[string]int32{}
	m.codes = m.pairCodes()
	if len(m.codes) < maxChoices {
		return fmt.Errorf("only %d single-token choice codes in this tokenizer", len(m.codes))
	}
	all := append(append([]string{}, noulCodes...), m.codes...)
	for d := range 10 {
		all = append(all, fmt.Sprint(d))
	}
	for _, code := range all {
		for _, form := range spellings(code) {
			m.singleToken(form)
		}
	}
	return nil
}

// singleToken is the one token text encodes to, or -1 when it takes several.
func (m *Model) singleToken(text string) int32 {
	if id, ok := m.single[text]; ok {
		return id
	}
	id := int32(-1)
	if ids := m.tok.Encode(text, false); len(ids) == 1 {
		id = ids[0]
	}
	m.single[text] = id
	return id
}

// pairCodes is A to Z, then the two-letter uppercase codes that are one token
// both bare and after a space, in alphabetical order, up to 255.
func (m *Model) pairCodes() []string {
	var codes []string
	for c := 'A'; c <= 'Z'; c++ {
		codes = append(codes, string(c))
	}
	for a := 'A'; a <= 'Z'; a++ {
		for b := 'A'; b <= 'Z'; b++ {
			code := string(a) + string(b)
			if m.singleToken(code) >= 0 && m.singleToken(" "+code) >= 0 {
				codes = append(codes, code)
			}
		}
	}
	if len(codes) > maxChoices {
		codes = codes[:maxChoices]
	}
	return codes
}

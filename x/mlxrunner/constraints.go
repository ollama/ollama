package mlxrunner

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"path/filepath"
	"slices"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/constraint"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

type requestConstraint interface {
	VocabSize() int
	Fill() ([]int32, bool, error)
	Accept(int32) error
	Close()
}

type constraintSpec struct {
	kind   constraint.Kind
	source string
}

const (
	maxConstraintSchemaBytes  = 1 << 20
	maxConstraintSchemaDepth  = 128
	maxConstraintSchemaTokens = 1 << 12
)

func parseConstraint(format json.RawMessage) (*constraintSpec, error) {
	if len(format) > 0 {
		if len(format) > maxConstraintSchemaBytes {
			return nil, fmt.Errorf("invalid format: input is %d bytes; limit is %d", len(format), maxConstraintSchemaBytes)
		}
		switch string(format) {
		case `null`, `""`:
			return nil, nil
		case `"json"`:
			return &constraintSpec{kind: constraint.JSON}, nil
		default:
			if format[0] != '{' {
				return nil, errors.New("invalid format: expected \"json\" or a valid JSON Schema object")
			}
			if err := validateConstraintSchema(format); err != nil {
				return nil, fmt.Errorf("invalid JSON Schema: %w", err)
			}
			return &constraintSpec{kind: constraint.JSONSchema, source: string(format)}, nil
		}
	}
	return nil, nil
}

func validateConstraintSchema(schema []byte) error {
	if len(schema) > maxConstraintSchemaBytes {
		return fmt.Errorf("schema is %d bytes; limit is %d", len(schema), maxConstraintSchemaBytes)
	}
	if !utf8.Valid(schema) {
		return errors.New("schema is not valid UTF-8")
	}

	decoder := json.NewDecoder(bytes.NewReader(schema))
	decoder.UseNumber()
	first, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if first != json.Delim('{') {
		return errors.New("schema must be a JSON object")
	}

	tokens, depth := 1, 1
	for depth > 0 {
		token, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				return errors.New("unexpected end of JSON")
			}
			return fmt.Errorf("invalid JSON: %w", err)
		}
		tokens++
		if tokens > maxConstraintSchemaTokens {
			return fmt.Errorf("schema contains more than %d JSON tokens", maxConstraintSchemaTokens)
		}

		if delim, ok := token.(json.Delim); ok {
			switch delim {
			case '{', '[':
				depth++
				if depth > maxConstraintSchemaDepth {
					return fmt.Errorf("schema nesting exceeds %d levels", maxConstraintSchemaDepth)
				}
			case '}', ']':
				depth--
			}
		}
	}

	if _, err := decoder.Token(); err != io.EOF {
		if err != nil {
			return fmt.Errorf("invalid JSON after schema object: %w", err)
		}
		return errors.New("schema contains more than one JSON value")
	}
	return nil
}

func (r *Runner) prepareConstraint(request *Request) error {
	spec, err := parseConstraint(request.Format)
	if err != nil || spec == nil {
		return err
	}
	if r.constraints == nil {
		message := "structured output is unavailable"
		if r.constraintErr != nil {
			message += ": " + r.constraintErr.Error()
		}
		return api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: message}
	}
	request.Constraint, err = r.constraints.Compile(spec.kind, spec.source)
	if err != nil {
		return fmt.Errorf("invalid structured output constraint: %w", err)
	}
	return nil
}

func (r *Runner) loadConstraints(root *model.Root) {
	library, err := mlx.LoadedLibraryPath()
	if err != nil {
		r.constraintErr = err
		return
	}
	if err := constraint.Load(filepath.Dir(library)); err != nil {
		r.constraintErr = err
		slog.Warn("Structured output is unavailable", "error", err)
		return
	}

	configuredVocabSize, err := modelVocabSize(root)
	if err != nil {
		r.constraintErr = err
		slog.Warn("Structured output is unavailable", "error", err)
		return
	}
	vocabSize, err := constraintVocabSize(configuredVocabSize, r.Tokenizer.VocabSize())
	if err != nil {
		r.constraintErr = err
		slog.Warn("Structured output is unavailable", "error", err)
		return
	}
	pieces := make([]string, vocabSize)
	for id := range vocabSize {
		pieces[id] = r.Tokenizer.Decode([]int32{int32(id)})
	}
	stops := slices.DeleteFunc(slices.Clone(r.Tokenizer.EOSTokens()), func(id int32) bool {
		return id < 0 || int(id) >= vocabSize
	})
	r.constraints, err = constraint.NewModel(pieces, vocabSize, stops)
	if err != nil {
		r.constraintErr = err
		slog.Warn("Structured output is unavailable", "error", err)
		return
	}
	r.constraintErr = nil
	slog.Info("Structured output initialized", "vocab_size", vocabSize, "library", constraint.LoadedLibraryPath())
}

func modelVocabSize(root *model.Root) (int, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return 0, fmt.Errorf("read model vocabulary size: %w", err)
	}
	return parseModelVocabSize(data)
}

func parseModelVocabSize(data []byte) (int, error) {
	var config struct {
		VocabSize  int `json:"vocab_size"`
		TextConfig struct {
			VocabSize int `json:"vocab_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return 0, fmt.Errorf("read model vocabulary size: %w", err)
	}
	if config.VocabSize < 0 || config.TextConfig.VocabSize < 0 {
		return 0, fmt.Errorf("invalid negative model vocabulary size: %d/%d", config.VocabSize, config.TextConfig.VocabSize)
	}
	if config.VocabSize > 0 {
		return config.VocabSize, nil
	}
	return config.TextConfig.VocabSize, nil
}

const maxConstraintVocabSize = 1 << 20

func constraintVocabSize(configured, tokenizerSize int) (int, error) {
	if tokenizerSize <= 0 {
		return 0, fmt.Errorf("invalid tokenizer vocabulary size %d", tokenizerSize)
	}
	if configured == 0 {
		configured = tokenizerSize
	}
	if configured < tokenizerSize {
		return 0, fmt.Errorf("model vocabulary size %d is smaller than tokenizer vocabulary size %d", configured, tokenizerSize)
	}
	if configured > maxConstraintVocabSize {
		return 0, fmt.Errorf("model vocabulary size %d exceeds structured output limit %d", configured, maxConstraintVocabSize)
	}
	return configured, nil
}

func (r *Runner) Close() {
	if r.constraints != nil {
		r.constraints.Close()
		r.constraints = nil
	}
}

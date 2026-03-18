package ortrunner

import (
	"context"
	"log/slog"
	"time"

	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Generate performs streaming token generation for a completion request.
func (r *Runner) Generate(ctx context.Context, req Request) error {
	maxTokens := req.Options.MaxTokens
	if maxTokens == 0 {
		maxTokens = req.Options.NumPredict
	}
	if maxTokens == 0 {
		maxTokens = 2048
	}

	params, err := oga.NewGeneratorParams(r.model)
	if err != nil {
		return err
	}
	defer params.Close()

	// Set search parameters
	if req.Options.Temperature > 0 {
		if err := params.SetNumber("temperature", float64(req.Options.Temperature)); err != nil {
			slog.Warn("failed to set temperature", "error", err)
		}
		if err := params.SetBool("do_sample", true); err != nil {
			slog.Warn("failed to set do_sample", "error", err)
		}
	}
	if req.Options.TopP > 0 {
		if err := params.SetNumber("top_p", float64(req.Options.TopP)); err != nil {
			slog.Warn("failed to set top_p", "error", err)
		}
	}
	if req.Options.TopK > 0 {
		if err := params.SetNumber("top_k", float64(req.Options.TopK)); err != nil {
			slog.Warn("failed to set top_k", "error", err)
		}
	}
	if err := params.SetNumber("max_length", float64(maxTokens)); err != nil {
		slog.Warn("failed to set max_length", "error", err)
	}

	gen, err := oga.NewGenerator(r.model, params)
	if err != nil {
		return err
	}
	defer gen.Close()

	// Append the prompt tokens
	if err := gen.AppendTokenSequencesFromEncoding(r.tokenizer, req.Prompt); err != nil {
		return err
	}

	stream, err := oga.NewTokenStream(r.tokenizer)
	if err != nil {
		return err
	}
	defer stream.Close()

	promptStart := time.Now()
	promptTokens := 0

	// Count prompt tokens
	if tokens, err := r.tokenizer.Encode(req.Prompt); err == nil {
		promptTokens = len(tokens)
	}

	// Generate first token (includes prompt processing time)
	if gen.IsDone() {
		req.Responses <- CompletionResponse{Done: true}
		return nil
	}
	if err := gen.GenerateNextToken(); err != nil {
		return err
	}
	promptDuration := time.Since(promptStart)

	genStart := time.Now()
	evalCount := 0

	// Process first generated token
	nextTokens, err := gen.GetNextTokens()
	if err != nil {
		return err
	}
	if len(nextTokens) > 0 {
		text, err := stream.Decode(nextTokens[0])
		if err != nil {
			return err
		}
		if text != "" {
			evalCount++
			select {
			case req.Responses <- CompletionResponse{Content: text}:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	// Generate remaining tokens
	for !gen.IsDone() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := gen.GenerateNextToken(); err != nil {
			return err
		}

		nextTokens, err := gen.GetNextTokens()
		if err != nil {
			return err
		}

		if len(nextTokens) == 0 {
			continue
		}

		text, err := stream.Decode(nextTokens[0])
		if err != nil {
			return err
		}

		evalCount++

		if text != "" {
			select {
			case req.Responses <- CompletionResponse{Content: text}:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	evalDuration := time.Since(genStart)

	// Send final done response with timing stats
	select {
	case req.Responses <- CompletionResponse{
		Done:               true,
		PromptEvalCount:    promptTokens,
		PromptEvalDuration: promptDuration,
		EvalCount:          evalCount,
		EvalDuration:       evalDuration,
	}:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

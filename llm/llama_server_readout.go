package llm

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

// scoreReadout evaluates each complete question through a trained linear head.
// Its calibrated logits already include the checkpoint's temperature.
func (s *llamaServerRunner) scoreReadout(ctx context.Context, input ScoreRequest) (ScoreResponse, error) {
	bad := func(message string) (ScoreResponse, error) {
		return ScoreResponse{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: message}
	}
	var images []api.ImageData
	for _, image := range input.Images {
		data, err := llamaServerMediaBytes(image)
		if err != nil {
			return bad(err.Error())
		}
		if !strings.HasPrefix(http.DetectContentType(data), "image/") {
			return bad("invalid decision image")
		}
		images = append(images, data)
	}
	var result ScoreResponse
	special := true
	for _, row := range input.Rows {
		if len(row.Candidates) < 1 || len(row.Candidates) > 255 {
			return bad("readout requires 1–255 options")
		}
		tokens, err := s.tokenize(ctx, row.Prompt, false, &special)
		if err != nil {
			return ScoreResponse{}, err
		}
		if len(tokens) == 0 || len(tokens) > input.MaxTokens {
			return bad("decision prompt exceeds the model context (input is never truncated)")
		}
		imagePosition := 0
		if len(images) > 0 {
			prefix, err := s.tokenize(ctx, row.ImagePrefix, false, &special)
			if err != nil {
				return ScoreResponse{}, err
			}
			if len(prefix) == 0 || len(prefix) >= len(tokens) || !slices.Equal(prefix, tokens[:len(prefix)]) {
				return bad("invalid image prefix")
			}
			imagePosition = len(prefix)
		}
		request := struct {
			Input         []int           `json:"input"`
			Options       int             `json:"score_options"`
			Images        []api.ImageData `json:"images,omitempty"`
			ImagePosition int             `json:"image_position,omitempty"`
		}{tokens, len(row.Candidates), images, imagePosition}
		var response []struct {
			Logits [][]float32 `json:"logits"`
			Tokens int         `json:"tokens_evaluated"`
		}
		if err := s.scoreRequest(ctx, "/embedding", request, &response); err != nil {
			return ScoreResponse{}, err
		}
		if len(response) != 1 || len(response[0].Logits) != 1 || len(response[0].Logits[0]) != len(row.Candidates) ||
			(len(images) == 0 && response[0].Tokens != len(tokens)) || (len(images) > 0 && response[0].Tokens <= len(tokens)) || response[0].Tokens > input.MaxTokens {
			return ScoreResponse{}, fmt.Errorf("decision runner did not score the complete request")
		}
		result.Logits = append(result.Logits, response[0].Logits[0])
		result.InputTokens += response[0].Tokens
	}
	return result, nil
}

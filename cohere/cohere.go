package cohere

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"strings"

	"golang.org/x/image/webp"

	"github.com/ollama/ollama/api"
)

func init() {
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)
}

type ErrorResponse struct {
	Message string `json:"message"`
}

type EmbedRequest struct {
	Model           string       `json:"model"`
	InputType       string       `json:"input_type,omitempty"`
	Texts           []string     `json:"texts,omitempty"`
	Images          []string     `json:"images,omitempty"`
	Inputs          []EmbedInput `json:"inputs,omitempty"`
	OutputDimension int          `json:"output_dimension,omitempty"`
	EmbeddingTypes  []string     `json:"embedding_types,omitempty"`
	Truncate        string       `json:"truncate,omitempty"`
}

type EmbedInput struct {
	Content []EmbedContent `json:"content"`
}

type EmbedContent struct {
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
	ImageURL *EmbedImageURL `json:"image_url,omitempty"`
}

type EmbedImageURL struct {
	URL string `json:"url"`
}

type EmbedResponse struct {
	ID           string          `json:"id"`
	Embeddings   EmbeddingTypes  `json:"embeddings"`
	Texts        []string        `json:"texts,omitempty"`
	Images       []ImageMetadata `json:"images,omitempty"`
	Meta         Meta            `json:"meta"`
	ResponseType string          `json:"response_type"`
}

type EmbeddingTypes struct {
	Float  [][]float32 `json:"float,omitempty"`
	Base64 []string    `json:"base64,omitempty"`
}

type ImageMetadata struct {
	Width    int    `json:"width,omitempty"`
	Height   int    `json:"height,omitempty"`
	Format   string `json:"format,omitempty"`
	BitDepth int    `json:"bit_depth,omitempty"`
}

type Meta struct {
	APIVersion  APIVersion  `json:"api_version"`
	BilledUnits BilledUnits `json:"billed_units"`
	Tokens      any         `json:"tokens"`
	Warnings    any         `json:"warnings"`
}

type APIVersion struct {
	Version        string `json:"version"`
	IsDeprecated   any    `json:"is_deprecated"`
	IsExperimental any    `json:"is_experimental"`
}

type BilledUnits struct {
	InputTokens     *int `json:"input_tokens"`
	OutputTokens    any  `json:"output_tokens"`
	SearchUnits     any  `json:"search_units"`
	Classifications any  `json:"classifications"`
	Images          *int `json:"images"`
}

func NewError(message string) ErrorResponse {
	return ErrorResponse{Message: message}
}

func FromEmbedRequest(r EmbedRequest) (api.EmbedRequest, error) {
	if len(r.Texts) > 0 && (len(r.Images) > 0 || len(r.Inputs) > 0) {
		return api.EmbedRequest{}, errors.New("texts cannot be combined with images or inputs")
	}
	if len(r.Images) > 0 && len(r.Inputs) > 0 {
		return api.EmbedRequest{}, errors.New("images cannot be combined with inputs")
	}

	req := api.EmbedRequest{
		Model:      r.Model,
		Dimensions: r.OutputDimension,
	}

	switch strings.ToUpper(strings.TrimSpace(r.Truncate)) {
	case "", "END":
		req.Truncate = ptr(true)
	case "NONE":
		req.Truncate = ptr(false)
	case "START":
		return api.EmbedRequest{}, errors.New("truncate=START is not supported")
	default:
		return api.EmbedRequest{}, fmt.Errorf("invalid truncate value: %s", r.Truncate)
	}

	switch {
	case len(r.Texts) > 0:
		if len(r.Texts) == 1 {
			req.Input = r.Texts[0]
		} else {
			req.Input = stringsToAny(r.Texts)
		}
	case len(r.Images) > 0:
		req.Inputs = make([]api.EmbedInput, 0, len(r.Images))
		for _, raw := range r.Images {
			img, err := decodeImageURL(raw)
			if err != nil {
				return api.EmbedRequest{}, err
			}
			req.Inputs = append(req.Inputs, api.EmbedInput{Image: img})
		}
	case len(r.Inputs) > 0:
		req.Inputs = make([]api.EmbedInput, 0, len(r.Inputs))
		for _, in := range r.Inputs {
			parsed, err := parseInput(in)
			if err != nil {
				return api.EmbedRequest{}, err
			}
			req.Inputs = append(req.Inputs, parsed)
		}
	default:
		return api.EmbedRequest{}, errors.New("must provide texts, images, or inputs")
	}

	return req, nil
}

func ToEmbedResponse(r EmbedRequest, resp api.EmbedResponse) (EmbedResponse, error) {
	types := normalizedEmbeddingTypes(r.EmbeddingTypes)

	out := EmbedResponse{
		ID:         "embed-response",
		Embeddings: EmbeddingTypes{},
		Meta: Meta{
			APIVersion:  APIVersion{Version: "2"},
			BilledUnits: billedUnits(resp.PromptEvalCount, imageCount(r)),
		},
		ResponseType: "embeddings_by_type",
	}

	if containsType(types, "float") {
		out.Embeddings.Float = resp.Embeddings
	}
	if containsType(types, "base64") {
		out.Embeddings.Base64 = make([]string, 0, len(resp.Embeddings))
		for _, e := range resp.Embeddings {
			out.Embeddings.Base64 = append(out.Embeddings.Base64, floatsToBase64(e))
		}
	}

	if len(r.Texts) > 0 {
		out.Texts = append([]string(nil), r.Texts...)
	} else if len(r.Inputs) > 0 {
		for _, in := range r.Inputs {
			text := strings.TrimSpace(extractText(in))
			if text != "" {
				out.Texts = append(out.Texts, text)
			}
		}
	}

	images, err := imageMetadata(r)
	if err != nil {
		return EmbedResponse{}, err
	}
	out.Images = images

	return out, nil
}

func ValidateEmbeddingTypes(types []string) error {
	for _, t := range normalizedEmbeddingTypes(types) {
		if t != "float" && t != "base64" {
			return fmt.Errorf("embedding type %q is not supported", t)
		}
	}
	return nil
}

func parseInput(in EmbedInput) (api.EmbedInput, error) {
	var out api.EmbedInput
	var texts []string

	for _, part := range in.Content {
		switch part.Type {
		case "text":
			texts = append(texts, part.Text)
		case "image_url":
			if part.ImageURL == nil {
				return api.EmbedInput{}, errors.New("image_url content requires image_url.url")
			}
			if len(out.Image) > 0 {
				return api.EmbedInput{}, errors.New("only one image per input is supported")
			}
			img, err := decodeImageURL(part.ImageURL.URL)
			if err != nil {
				return api.EmbedInput{}, err
			}
			out.Image = img
		default:
			return api.EmbedInput{}, fmt.Errorf("unsupported content type: %s", part.Type)
		}
	}

	out.Text = strings.Join(texts, "\n")
	if out.Text == "" && len(out.Image) == 0 {
		return api.EmbedInput{}, errors.New("input content cannot be empty")
	}

	return out, nil
}

func extractText(in EmbedInput) string {
	var texts []string
	for _, part := range in.Content {
		if part.Type == "text" && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func imageMetadata(r EmbedRequest) ([]ImageMetadata, error) {
	var raws []string
	raws = append(raws, r.Images...)
	for _, in := range r.Inputs {
		for _, part := range in.Content {
			if part.Type == "image_url" && part.ImageURL != nil {
				raws = append(raws, part.ImageURL.URL)
			}
		}
	}

	images := make([]ImageMetadata, 0, len(raws))
	for _, raw := range raws {
		img, err := decodeImageURL(raw)
		if err != nil {
			return nil, err
		}
		cfg, format, err := image.DecodeConfig(bytes.NewReader(img))
		if err != nil {
			return nil, err
		}
		images = append(images, ImageMetadata{
			Width:  cfg.Width,
			Height: cfg.Height,
			Format: format,
		})
	}

	return images, nil
}

func imageCount(r EmbedRequest) int {
	count := len(r.Images)
	for _, in := range r.Inputs {
		for _, part := range in.Content {
			if part.Type == "image_url" && part.ImageURL != nil {
				count++
			}
		}
	}
	return count
}

func billedUnits(promptTokens int, images int) BilledUnits {
	var inputTokens *int
	if promptTokens > 0 {
		inputTokens = &promptTokens
	}

	var billedImages *int
	if images > 0 {
		billedImages = &images
	}

	return BilledUnits{
		InputTokens: inputTokens,
		Images:      billedImages,
	}
}

func normalizedEmbeddingTypes(types []string) []string {
	if len(types) == 0 {
		return []string{"float"}
	}

	out := make([]string, 0, len(types))
	seen := map[string]struct{}{}
	for _, t := range types {
		key := strings.ToLower(strings.TrimSpace(t))
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	return out
}

func containsType(types []string, want string) bool {
	for _, t := range types {
		if t == want {
			return true
		}
	}
	return false
}

func stringsToAny(values []string) []any {
	out := make([]any, 0, len(values))
	for _, v := range values {
		out = append(out, v)
	}
	return out
}

func ptr[T any](v T) *T {
	return &v
}

func floatsToBase64(floats []float32) string {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, floats)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func decodeImageURL(url string) (api.ImageData, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return nil, errors.New("image URLs are not currently supported, please use base64 encoded data instead")
	}

	types := []string{"jpeg", "jpg", "png", "webp", "gif"}

	if strings.HasPrefix(url, "data:;base64,") {
		url = strings.TrimPrefix(url, "data:;base64,")
	} else {
		valid := false
		for _, t := range types {
			prefix := "data:image/" + t + ";base64,"
			if strings.HasPrefix(url, prefix) {
				url = strings.TrimPrefix(url, prefix)
				valid = true
				break
			}
		}
		if !valid {
			return nil, errors.New("invalid image input")
		}
	}

	img, err := base64.StdEncoding.DecodeString(url)
	if err != nil {
		return nil, errors.New("invalid image input")
	}
	return img, nil
}

package launch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// Gemini implements Runner for native Gemini API integration.
type Gemini struct{}

func (g *Gemini) String() string { return "Gemini API" }

// Gemini API request structures
type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
}

type geminiContent struct {
	Role    string          `json:"role"`
	Parts   []geminiPart    `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

// Gemini API response structures
type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

func (g *Gemini) Run(model string, args []string) error {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("GEMINI_API_KEY environment variable is required")
	}

	if model == "" {
		model = "gemini-1.5-flash"
	}

	// Construct prompt from args
	prompt := strings.Join(args, " ")
	if prompt == "" {
		return fmt.Errorf("no prompt provided")
	}

	// Prepare request body
	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Role:  "user",
				Parts: []geminiPart{{Text: prompt}},
			},
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %v", err)
	}

	// Use streamGenerateContent for a better experience
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?key=%s", model, apiKey)

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API returned non-200 status: %d, body: %s", resp.StatusCode, string(body))
	}

	// Gemini streamGenerateContent returns a JSON array of response objects
	// We need to parse the stream and extract the text content
	decoder := json.NewDecoder(resp.Body)

	// The API returns a JSON array [ {candidates: [...]}, {candidates: [...]}, ... ]
	// We need to handle the start of the array '['
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("failed to read start of stream: %v", err)
	}
	if token != json.Delim('[') {
		return fmt.Errorf("expected JSON array from Gemini API")
	}

	for decoder.More() {
		var chunk geminiResponse
		if err := decoder.Decode(&chunk); err != nil {
			return fmt.Errorf("failed to decode stream chunk: %v", err)
		}
		for _, cand := range chunk.Candidates {
			for _, part := range cand.Content.Parts {
				fmt.Print(part.Text)
			}
		}
	}

	return nil
}

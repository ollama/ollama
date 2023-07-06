package api

import (
	"fmt"
	"net/http"
	"strings"
)

type Error struct {
	Code    int32  `json:"code"`
	Message string `json:"message"`
}

func (e Error) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("%d %v", e.Code, strings.ToLower(http.StatusText(int(e.Code))))
	}
	return e.Message
}

type PullRequest struct {
	Model string `json:"model"`
}

type PullResponse struct {
	Response string `json:"response"`
}

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type GenerateResponse struct {
	Response string `json:"response"`
}

type TokenResponse struct {
	Choices []TokenResponseChoice `json:"choices"`
}

type TokenResponseChoice struct {
	Text string `json:"text"`
}

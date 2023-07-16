package server

import (
	"github.com/jmorganca/ollama/api"
)

type Model struct {
	Name             string `json:"name"`
	ModelPath        string
	Prompt           string
	Options          api.Options
	DisplayName      string `json:"display_name"`
	Parameters       string `json:"parameters"`
	URL              string `json:"url"`
	ShortDescription string `json:"short_description"`
	Description      string `json:"description"`
	PublishedBy      string `json:"published_by"`
	OriginalAuthor   string `json:"original_author"`
	OriginalURL      string `json:"original_url"`
	License          string `json:"license"`
}


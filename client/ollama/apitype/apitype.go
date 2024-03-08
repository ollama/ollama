package apitype

import "time"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Model struct {
	Ref        string `json:"ref"`
	Digest     string `json:"digest"`
	Size       int64  `json:"size"`
	ModifiedAt int64  `json:"modified"`
}

func (m Model) Modifed() time.Time {
	return time.Unix(0, m.ModifiedAt)
}

type PushRequest struct {
	Name     string `json:"name"` // Ref is the official term, "name" is for backward compatibility with exiting clients.
	Insecure bool   `json:"insecure"`
	Stream   bool   `json:"stream"`
}

type PushStatus struct {
	Status string `json:"status"`
	Digest string `json:"digest"`
	Total  int64  `json:"total"`
}

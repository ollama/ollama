package decision

import (
	"fmt"
	"net/url"
	"strings"
)

// PointerHeadFamily marks a decision model whose probabilities come from a
// pointer head, not from next-token letter logits. Letter scoring must not be
// used as a fallback: it returns a confident distribution over the wrong tokens.
const PointerHeadFamily = "pointer-head"

// IsPointerFamily reports whether a model config selects the pointer-head runner.
func IsPointerFamily(family string, families []string) bool {
	if family == PointerHeadFamily {
		return true
	}
	for _, item := range families {
		if item == PointerHeadFamily {
			return true
		}
	}
	return false
}

// PointerRunnerEndpoint resolves OLLAMA_POINTER_RUNNER to a loopback
// POST /v1/systemone URL. An empty value returns an empty endpoint and no
// error; the caller refuses the request instead of letter-scoring it.
func PointerRunnerEndpoint(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || parsed.User != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", fmt.Errorf("OLLAMA_POINTER_RUNNER must be an http(s) loopback URL")
	}
	switch parsed.Hostname() {
	case "127.0.0.1", "localhost", "::1":
	default:
		return "", fmt.Errorf("OLLAMA_POINTER_RUNNER must point at a loopback address")
	}
	path := strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(path, "/v1/systemone") {
		path += "/v1/systemone"
	}
	parsed.Path = path
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

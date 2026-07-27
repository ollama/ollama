package server

import "testing"

func TestRouteToAction(t *testing.T) {
	cases := map[string]string{
		"/api/chat":            "chat",
		"/v1/chat/completions": "chat",
		"/api/embed":           "embed",
		"/v1/embeddings":       "embed",
		"/api/generate":        "generate",
		"/api/show":            "show",
		"/v1/models":           "models",
		"/":                    "/",
	}

	for input, want := range cases {
		got := routeToAction(input)
		if got != want {
			t.Fatalf("routeToAction(%q) = %q, want %q", input, got, want)
		}
	}
}

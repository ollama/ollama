//go:build windows || darwin

package ui

import (
	"html"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/app/ui/responses"
)

// The library page is server-rendered HTML with no JSON API, so the catalog is
// read from the markup. These patterns are deliberately narrow: each one keys
// off a stable class or label rather than positional structure, and a row that
// does not match is skipped instead of producing a half-populated entry.
var (
	libraryRowRe     = regexp.MustCompile(`(?s)<li\s+class="flex items-baseline border-b border-neutral-200 py-6">(.*?)</li>`)
	libraryLinkRe    = regexp.MustCompile(`href="/library/([a-z0-9._\-]+)"`)
	libraryDescRe    = regexp.MustCompile(`(?s)<p class="max-w-lg break-words text-neutral-800 text-md">(.*?)</p>`)
	libraryCapRe     = regexp.MustCompile(`(?s)<span\s+class="inline-flex items-center rounded-md bg-indigo-50[^"]*">([^<]+)</span>`)
	libraryVariantRe = regexp.MustCompile(`(?s)<span\s+class="inline-flex items-center rounded-md bg-\[#ddf4ff\][^"]*">([^<]+)</span>`)
	libraryPullsRe   = regexp.MustCompile(`(?s)<span\s*>([0-9.]+[KMB]?)</span>\s*<span class="hidden sm:flex">&nbsp;Pulls</span>`)
	libraryTagsRe    = regexp.MustCompile(`(?s)<span\s*>([0-9.]+[KMB]?)</span>\s*<span class="hidden sm:flex">&nbsp;Tags</span>`)
)

// parseLibrary extracts the model catalog from the library page markup.
func parseLibrary(r io.Reader) ([]responses.ExploreModel, error) {
	body, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	rows := libraryRowRe.FindAllStringSubmatch(string(body), -1)
	models := make([]responses.ExploreModel, 0, len(rows))

	for _, row := range rows {
		entry := row[1]

		name := ""
		if m := libraryLinkRe.FindStringSubmatch(entry); m != nil {
			name = m[1]
		}
		if name == "" {
			continue
		}

		model := responses.ExploreModel{Model: name}

		if m := libraryDescRe.FindStringSubmatch(entry); m != nil {
			model.Description = strings.TrimSpace(html.UnescapeString(m[1]))
		}

		for _, m := range libraryCapRe.FindAllStringSubmatch(entry, -1) {
			model.Capabilities = append(model.Capabilities, strings.TrimSpace(m[1]))
		}

		for _, m := range libraryVariantRe.FindAllStringSubmatch(entry, -1) {
			model.Variants = append(model.Variants, strings.TrimSpace(m[1]))
		}

		if m := libraryPullsRe.FindStringSubmatch(entry); m != nil {
			model.Pulls = m[1]
		}

		if m := libraryTagsRe.FindStringSubmatch(entry); m != nil {
			if n, err := strconv.Atoi(m[1]); err == nil {
				model.Tags = n
			}
		}

		models = append(models, model)
	}

	return models, nil
}

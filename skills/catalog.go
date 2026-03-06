package skills

import (
	"encoding/json"
	"errors"
	"os"
	"slices"
	"sort"
	"strings"
)

const envSkillCatalog = "OLLAMA_SKILL_CATALOG"

type CatalogEntry struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Source      string   `json:"source"`
	Tags        []string `json:"tags,omitempty"`
	Permissions []string `json:"permissions,omitempty"`
	Verified    bool     `json:"verified,omitempty"`
	UpdatedAt   string   `json:"updated_at,omitempty"`
}

type CatalogFilter struct {
	Query        string
	Tags         []string
	Permissions  []string
	VerifiedOnly bool
}

func SearchCatalog(filter CatalogFilter) ([]CatalogEntry, error) {
	catalog, err := loadCatalog()
	if err != nil {
		return nil, err
	}

	terms := strings.Fields(strings.ToLower(strings.TrimSpace(filter.Query)))
	tags := normalizeStringList(filter.Tags)
	perms, err := normalizePermissions(filter.Permissions)
	if err != nil {
		return nil, err
	}

	out := make([]CatalogEntry, 0, len(catalog))
	for _, entry := range catalog {
		entry.Name = strings.TrimSpace(entry.Name)
		entry.Description = strings.TrimSpace(entry.Description)
		entry.Source = strings.TrimSpace(entry.Source)
		entry.Tags = normalizeStringList(entry.Tags)
		entry.Permissions, err = normalizePermissions(entry.Permissions)
		if err != nil {
			return nil, err
		}
		if entry.Name == "" || entry.Source == "" {
			continue
		}
		if filter.VerifiedOnly && !entry.Verified {
			continue
		}
		if !entryMatchesTerms(entry, terms) {
			continue
		}
		if !entryMatchesTags(entry, tags) {
			continue
		}
		if !entryMatchesPermissions(entry, perms) {
			continue
		}
		out = append(out, entry)
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Verified != out[j].Verified {
			return out[i].Verified
		}
		return out[i].Name < out[j].Name
	})

	return out, nil
}

func loadCatalog() ([]CatalogEntry, error) {
	path := strings.TrimSpace(os.Getenv(envSkillCatalog))
	if path == "" {
		return slices.Clone(defaultCatalog), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var entries []CatalogEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, errors.New("catalog is empty")
	}
	return entries, nil
}

func entryMatchesTerms(entry CatalogEntry, terms []string) bool {
	if len(terms) == 0 {
		return true
	}

	index := strings.ToLower(strings.Join([]string{
		entry.Name,
		entry.Description,
		strings.Join(entry.Tags, " "),
		strings.Join(entry.Permissions, " "),
		entry.Source,
	}, " "))

	for _, term := range terms {
		if !strings.Contains(index, term) {
			return false
		}
	}
	return true
}

func entryMatchesTags(entry CatalogEntry, required []string) bool {
	if len(required) == 0 {
		return true
	}
	set := map[string]bool{}
	for _, tag := range entry.Tags {
		set[strings.ToLower(tag)] = true
	}
	for _, tag := range required {
		if !set[strings.ToLower(tag)] {
			return false
		}
	}
	return true
}

func entryMatchesPermissions(entry CatalogEntry, required []string) bool {
	if len(required) == 0 {
		return true
	}
	set := map[string]bool{}
	for _, permission := range entry.Permissions {
		set[permission] = true
	}
	for _, permission := range required {
		if !set[permission] {
			return false
		}
	}
	return true
}

var defaultCatalog = []CatalogEntry{
	{
		Name:        "git-helper",
		Description: "Assist with local git status, diffs, and branch hygiene.",
		Source:      "github.com/ollama/skills-git@v1.0.0",
		Tags:        []string{"dev", "git"},
		Permissions: []string{"filesystem.read", "filesystem.write"},
		Verified:    true,
		UpdatedAt:   "2026-03-01",
	},
	{
		Name:        "web-research",
		Description: "Fetch and summarize web sources with citations.",
		Source:      "github.com/ollama/skills-web@v1.2.0",
		Tags:        []string{"research", "web"},
		Permissions: []string{"network.fetch"},
		Verified:    true,
		UpdatedAt:   "2026-03-02",
	},
	{
		Name:        "ops-health",
		Description: "Read service logs and summarize incidents.",
		Source:      "github.com/community/ops-health@2b49b98",
		Tags:        []string{"ops", "monitoring"},
		Permissions: []string{"filesystem.read"},
		Verified:    false,
		UpdatedAt:   "2026-02-25",
	},
}

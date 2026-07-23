package cmd

import (
	"cmp"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

const localSemanticMemoryHeader = "Local semantic memory profile (stored only on this device):"

var memoryTokenPattern = regexp.MustCompile(`[a-z0-9]+`)

type localSemanticMemoryManager struct {
	dir          string
	maxFacts     int
	profileFacts int
}

func newLocalSemanticMemoryManager() (*localSemanticMemoryManager, error) {
	if !envconfig.LocalSemanticMemoryEnabled() {
		return nil, nil
	}

	dir := strings.TrimSpace(envconfig.LocalSemanticMemoryDir())
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		dir = filepath.Join(home, ".ollama", "llm_memory")
	}

	maxFacts := max(2, int(envconfig.LocalSemanticMemoryMaxFacts()))
	profileFacts := max(1, int(envconfig.LocalSemanticMemoryProfileFacts()))
	profileFacts = min(profileFacts, maxFacts)

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	return &localSemanticMemoryManager{
		dir:          dir,
		maxFacts:     maxFacts,
		profileFacts: profileFacts,
	}, nil
}

func (m *localSemanticMemoryManager) injectSystemMessage(msgs []api.Message) ([]api.Message, error) {
	for _, msg := range msgs {
		if msg.Role == "system" && strings.Contains(msg.Content, localSemanticMemoryHeader) {
			return msgs, nil
		}
	}

	facts, err := m.readLatestFacts(m.profileFacts)
	if err != nil {
		return nil, err
	}
	if len(facts) == 0 {
		return msgs, nil
	}

	var b strings.Builder
	b.WriteString(localSemanticMemoryHeader)
	b.WriteString("\n")
	for _, fact := range facts {
		b.WriteString("- ")
		b.WriteString(fact)
		b.WriteString("\n")
	}
	b.WriteString("Use this as probabilistic user context and defer to the latest user message when conflicts appear.")

	memMessage := api.Message{
		Role:    "system",
		Content: b.String(),
	}
	return append([]api.Message{memMessage}, msgs...), nil
}

func (m *localSemanticMemoryManager) captureUserFact(content string) error {
	fact := normalizeMemoryFact(content)
	if fact == "" {
		return nil
	}

	filename := fmt.Sprintf("fact-%d.txt", time.Now().UTC().UnixNano())
	path := filepath.Join(m.dir, filename)
	if err := os.WriteFile(path, []byte(fact), 0o600); err != nil {
		return err
	}

	return m.compressIfNeeded()
}

func normalizeMemoryFact(content string) string {
	content = strings.TrimSpace(content)
	if content == "" {
		return ""
	}

	fields := strings.Fields(content)
	if len(fields) < 3 {
		return ""
	}

	fact := strings.Join(fields, " ")
	if len(fact) > 280 {
		fact = strings.TrimSpace(fact[:280])
	}
	return fact
}

func (m *localSemanticMemoryManager) compressIfNeeded() error {
	files, err := m.listFactFiles()
	if err != nil {
		return err
	}

	for len(files) > m.maxFacts {
		left, right := selectMostSimilarFacts(files)
		if left == nil || right == nil {
			break
		}

		merged, err := mergeFactContents(left.path, right.path)
		if err != nil {
			return err
		}

		filename := fmt.Sprintf("fact-merged-%d.txt", time.Now().UTC().UnixNano())
		path := filepath.Join(m.dir, filename)
		if err := os.WriteFile(path, []byte(merged), 0o600); err != nil {
			return err
		}

		if err := os.Remove(left.path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err := os.Remove(right.path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		files, err = m.listFactFiles()
		if err != nil {
			return err
		}
	}

	return nil
}

func mergeFactContents(leftPath, rightPath string) (string, error) {
	leftBytes, err := os.ReadFile(leftPath)
	if err != nil {
		return "", err
	}
	rightBytes, err := os.ReadFile(rightPath)
	if err != nil {
		return "", err
	}

	left := strings.TrimSpace(string(leftBytes))
	right := strings.TrimSpace(string(rightBytes))
	if left == "" {
		return right, nil
	}
	if right == "" {
		return left, nil
	}
	if strings.EqualFold(left, right) {
		return left, nil
	}
	if strings.Contains(strings.ToLower(left), strings.ToLower(right)) {
		return left, nil
	}
	if strings.Contains(strings.ToLower(right), strings.ToLower(left)) {
		return right, nil
	}
	return left + "\n" + right, nil
}

type memoryFactFile struct {
	path    string
	modTime time.Time
	content string
}

func (m *localSemanticMemoryManager) listFactFiles() ([]memoryFactFile, error) {
	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return nil, err
	}

	out := make([]memoryFactFile, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".txt") {
			continue
		}

		path := filepath.Join(m.dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			return nil, err
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		content := strings.TrimSpace(string(contentBytes))
		if content == "" {
			continue
		}
		out = append(out, memoryFactFile{
			path:    path,
			modTime: info.ModTime(),
			content: content,
		})
	}

	slices.SortFunc(out, func(a, b memoryFactFile) int {
		return cmp.Compare(a.modTime.UnixNano(), b.modTime.UnixNano())
	})
	return out, nil
}

func (m *localSemanticMemoryManager) readLatestFacts(limit int) ([]string, error) {
	files, err := m.listFactFiles()
	if err != nil {
		return nil, err
	}

	if len(files) == 0 {
		return nil, nil
	}

	start := max(0, len(files)-limit)
	recent := files[start:]
	out := make([]string, 0, len(recent))
	for i := len(recent) - 1; i >= 0; i-- {
		out = append(out, recent[i].content)
	}
	return out, nil
}

func selectMostSimilarFacts(files []memoryFactFile) (*memoryFactFile, *memoryFactFile) {
	if len(files) < 2 {
		return nil, nil
	}

	bestScore := -1.0
	var left, right *memoryFactFile
	for i := 0; i < len(files)-1; i++ {
		for j := i + 1; j < len(files); j++ {
			score := memorySimilarity(files[i].content, files[j].content)
			if score > bestScore {
				bestScore = score
				left = &files[i]
				right = &files[j]
			}
		}
	}
	return left, right
}

func memorySimilarity(a, b string) float64 {
	aTokens := memoryTokens(a)
	bTokens := memoryTokens(b)
	if len(aTokens) == 0 || len(bTokens) == 0 {
		return 0
	}

	intersection := 0
	for token := range aTokens {
		if _, ok := bTokens[token]; ok {
			intersection++
		}
	}
	union := len(aTokens) + len(bTokens) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func memoryTokens(content string) map[string]struct{} {
	matches := memoryTokenPattern.FindAllString(strings.ToLower(content), -1)
	set := make(map[string]struct{}, len(matches))
	for _, match := range matches {
		if len(match) <= 2 {
			continue
		}
		set[match] = struct{}{}
	}
	return set
}

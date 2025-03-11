package readline

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/emirpasic/gods/v2/lists/arraylist"
)

type History struct {
	Enabled bool

	lines    *arraylist.List[string]
	limit    int
	pos      int
	filename string
}

func NewHistory() (*History, error) {
	h := &History{
		Enabled: true,
		lines:   arraylist.New[string](),
		limit:   100, // resizeme
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	path := filepath.Join(home, ".ollama", "history")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}

	h.filename = path

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDONLY, 0o600)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if line := strings.TrimSpace(scanner.Text()); len(line) > 0 {
			h.Add(line)
		}
	}

	return h, nil
}

func (h *History) Add(s string) {
	if latest, _ := h.lines.Get(h.Size() - 1); latest != s {
		h.lines.Add(s)
		h.Compact()
		_ = h.Save()
	}
	// always set position to the end
	h.pos = h.Size()
}

func (h *History) Compact() {
	if s := h.lines.Size(); s > h.limit {
		for range s - h.limit {
			h.lines.Remove(0)
		}
	}
}

func (h *History) Clear() {
	h.lines.Clear()
}

func (h *History) Prev() (line string) {
	if h.pos > 0 {
		h.pos -= 1
	}
	// return first line if at the beginning
	line, _ = h.lines.Get(h.pos)
	return line
}

func (h *History) Next() (line string) {
	if h.pos < h.lines.Size() {
		h.pos += 1
		line, _ = h.lines.Get(h.pos)
	}
	// return empty string if at the end
	return line
}

func (h *History) Size() int {
	return h.lines.Size()
}

func (h *History) Save() error {
	if !h.Enabled {
		return nil
	}

	f, err := os.CreateTemp(filepath.Dir(h.filename), "")
	if err != nil {
		return err
	}

	func() {
		defer f.Close()

		w := bufio.NewWriter(f)
		defer w.Flush()

		h.lines.Each(func(i int, line string) {
			fmt.Fprintln(w, line)
		})
	}()

	return os.Rename(f.Name(), h.filename)
}

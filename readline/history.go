package readline

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/emirpasic/gods/v2/lists/arraylist"
)

type History struct {
	Buf      *arraylist.List[string]
	Autosave bool
	Pos      int
	Limit    int
	Filename string
	Enabled  bool
}

func NewHistory() (*History, error) {
	h := &History{
		Buf:      arraylist.New[string](),
		Limit:    100, // resizeme
		Autosave: true,
		Enabled:  true,
	}

	err := h.Init()
	if err != nil {
		return nil, err
	}

	return h, nil
}

func (h *History) Init() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	path := filepath.Join(home, ".ollama", "history")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	h.Filename = path

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDONLY, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}

		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		h.Add(line)
	}

	return nil
}

func (h *History) Add(s string) {
	h.Buf.Add(s)
	h.Compact()
	h.Pos = h.Size()
	if h.Autosave {
		_ = h.Save()
	}
}

func (h *History) Compact() {
	s := h.Buf.Size()
	if s > h.Limit {
		for range s - h.Limit {
			h.Buf.Remove(0)
		}
	}
}

func (h *History) Clear() {
	h.Buf.Clear()
}

func (h *History) Prev() (line string) {
	if h.Pos > 0 {
		h.Pos -= 1
	}
	line, _ = h.Buf.Get(h.Pos)
	return line
}

func (h *History) Next() (line string) {
	if h.Pos < h.Buf.Size() {
		h.Pos += 1
		line, _ = h.Buf.Get(h.Pos)
	}
	return line
}

func (h *History) Size() int {
	return h.Buf.Size()
}

func (h *History) Save() error {
	if !h.Enabled {
		return nil
	}

	tmpFile := h.Filename + ".tmp"

	f, err := os.OpenFile(tmpFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := bufio.NewWriter(f)
	for cnt := range h.Size() {
		line, _ := h.Buf.Get(cnt)
		fmt.Fprintln(buf, line)
	}
	buf.Flush()
	f.Close()

	if err = os.Rename(tmpFile, h.Filename); err != nil {
		return err
	}

	return nil
}

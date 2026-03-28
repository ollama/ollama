package readline

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/emirpasic/gods/v2/lists/arraylist"
)

type History struct {
	Buf            *arraylist.List[string]
	Autosave       bool
	Pos            int
	Limit          int
	Filename       string
	Enabled        bool
	Lock           sync.Mutex
	FileDescriptor *os.File
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
	h.Lock.Lock()
	defer h.Lock.Unlock()

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	path := filepath.Join(home, ".ollama", "history")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	h.Filename = path

	if err := h.closeLocked(); err != nil {
		return err
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDONLY, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	h.FileDescriptor = f

	h.Buf.Clear()
	h.Pos = 0

	if h.FileDescriptor == nil {
		return nil
	}

	r := bufio.NewReader(h.FileDescriptor)
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

		h.addLocked(line)
	}

	return nil
}

func (h *History) Add(s string) {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	h.addLocked(s)
	if h.Autosave {
		_ = h.saveLocked()
	}
}

func (h *History) Compact() {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	h.compactLocked()
}

func (h *History) compactLocked() {
	s := h.Buf.Size()
	if s > h.Limit {
		for range s - h.Limit {
			h.Buf.Remove(0)
		}
	}
}

func (h *History) Clear() {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	h.Buf.Clear()
}

func (h *History) Prev() (line string) {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	if h.Pos > 0 {
		h.Pos -= 1
	}
	line, _ = h.Buf.Get(h.Pos)
	return line
}

func (h *History) Next() (line string) {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	if h.Pos < h.Buf.Size() {
		h.Pos += 1
		line, _ = h.Buf.Get(h.Pos)
	}
	return line
}

func (h *History) Size() int {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	return h.Buf.Size()
}

func (h *History) Save() error {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	return h.saveLocked()
}

func (h *History) saveLocked() error {
	if !h.Enabled {
		return nil
	}
	if h.Filename == "" {
		return errors.New("history filename not initialized")
	}

	tmpFile := h.Filename + ".tmp"

	f, err := os.OpenFile(tmpFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}

	buf := bufio.NewWriter(f)
	for cnt := range h.Buf.Size() {
		line, _ := h.Buf.Get(cnt)
		if _, err := fmt.Fprintln(buf, line); err != nil {
			return errors.Join(err, closeFile(f))
		}
	}
	if err := buf.Flush(); err != nil {
		return errors.Join(err, closeFile(f))
	}
	if err := closeFile(f); err != nil {
		return err
	}

	if err := h.closeLocked(); err != nil {
		return err
	}

	if err = os.Rename(tmpFile, h.Filename); err != nil {
		return err
	}

	h.FileDescriptor, err = os.OpenFile(h.Filename, os.O_RDONLY, 0o600)
	if err != nil {
		return err
	}

	return nil
}

func (h *History) addLocked(s string) {
	h.Buf.Add(s)
	h.compactLocked()
	h.Pos = h.Buf.Size()
}

func (h *History) closeLocked() error {
	if h.FileDescriptor == nil {
		return nil
	}

	err := h.FileDescriptor.Close()
	h.FileDescriptor = nil
	return err
}

func (h *History) setEnabled(enabled bool) {
	h.Lock.Lock()
	defer h.Lock.Unlock()

	h.Enabled = enabled
}

func closeFile(f *os.File) error {
	if f == nil {
		return nil
	}
	return f.Close()
}

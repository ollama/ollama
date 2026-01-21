package parser

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/mod/semver"
	"golang.org/x/sync/errgroup"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"

	"github.com/ollama/ollama/api"
)

var ErrModelNotFound = errors.New("no Modelfile or safetensors files found")

type Modelfile struct {
	Commands []Command
}

func (f Modelfile) String() string {
	var sb strings.Builder
	for _, cmd := range f.Commands {
		fmt.Fprintln(&sb, cmd.String())
	}

	return sb.String()
}

var deprecatedParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

// CreateRequest creates a new *api.CreateRequest from an existing Modelfile
func (f Modelfile) CreateRequest(relativeDir string) (*api.CreateRequest, error) {
	req := &api.CreateRequest{}

	var messages []api.Message
	var licenses []string
	params := make(map[string]any)

	for _, c := range f.Commands {
		switch c.Name {
		case "model":
			path, err := expandPath(c.Args, relativeDir)
			if err != nil {
				return nil, err
			}

			digestMap, err := fileDigestMap(path)
			if errors.Is(err, os.ErrNotExist) {
				req.From = c.Args
				continue
			} else if err != nil {
				return nil, err
			}

			if req.Files == nil {
				req.Files = digestMap
			} else {
				for k, v := range digestMap {
					req.Files[k] = v
				}
			}
		case "adapter":
			path, err := expandPath(c.Args, relativeDir)
			if err != nil {
				return nil, err
			}

			digestMap, err := fileDigestMap(path)
			if err != nil {
				return nil, err
			}

			req.Adapters = digestMap
		case "template":
			req.Template = c.Args
		case "system":
			req.System = c.Args
		case "license":
			licenses = append(licenses, c.Args)
		case "renderer":
			req.Renderer = c.Args
		case "parser":
			req.Parser = c.Args
		case "requires":
			// golang.org/x/mod/semver requires "v" prefix
			requires := c.Args
			if !strings.HasPrefix(requires, "v") {
				requires = "v" + requires
			}
			if !semver.IsValid(requires) {
				return nil, fmt.Errorf("requires must be a valid semver (e.g. 0.14.0)")
			}
			req.Requires = strings.TrimPrefix(requires, "v")
		case "message":
			role, msg, _ := strings.Cut(c.Args, ": ")
			messages = append(messages, api.Message{Role: role, Content: msg})
		default:
			if slices.Contains(deprecatedParameters, c.Name) {
				fmt.Printf("warning: parameter %s is deprecated\n", c.Name)
				break
			}

			ps, err := api.FormatParams(map[string][]string{c.Name: {c.Args}})
			if err != nil {
				return nil, err
			}

			for k, v := range ps {
				if ks, ok := params[k].([]string); ok {
					params[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					params[k] = vs
				} else {
					params[k] = v
				}
			}
		}
	}

	if len(params) > 0 {
		req.Parameters = params
	}
	if len(messages) > 0 {
		req.Messages = messages
	}
	if len(licenses) > 0 {
		req.License = licenses
	}

	return req, nil
}

func fileDigestMap(path string) (map[string]string, error) {
	fl := make(map[string]string)

	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	var files []string
	if fi.IsDir() {
		fs, err := filesForModel(path)
		if err != nil {
			return nil, err
		}

		for _, f := range fs {
			f, err := filepath.EvalSymlinks(f)
			if err != nil {
				return nil, err
			}

			rel, err := filepath.Rel(path, f)
			if err != nil {
				return nil, err
			}

			if !filepath.IsLocal(rel) {
				return nil, fmt.Errorf("insecure path: %s", rel)
			}

			files = append(files, f)
		}
	} else {
		files = []string{path}
	}

	var mu sync.Mutex
	var g errgroup.Group
	g.SetLimit(max(runtime.GOMAXPROCS(0)-1, 1))
	for _, f := range files {
		g.Go(func() error {
			digest, err := digestForFile(f)
			if err != nil {
				return err
			}

			mu.Lock()
			defer mu.Unlock()
			fl[f] = digest
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return fl, nil
}

func digestForFile(filename string) (string, error) {
	filepath, err := filepath.EvalSymlinks(filename)
	if err != nil {
		return "", err
	}

	bin, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, bin); err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", hash.Sum(nil)), nil
}

func filesForModel(path string) ([]string, error) {
	detectContentType := func(path string) (string, error) {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		defer f.Close()

		var b bytes.Buffer
		b.Grow(512)

		if _, err := io.CopyN(&b, f, 512); err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}

		contentType, _, _ := strings.Cut(http.DetectContentType(b.Bytes()), ";")
		return contentType, nil
	}

	glob := func(pattern, contentType string) ([]string, error) {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, err
		}

		for _, match := range matches {
			if ct, err := detectContentType(match); err != nil {
				return nil, err
			} else if len(contentType) > 0 && ct != contentType {
				return nil, fmt.Errorf("invalid content type: expected %s for %s", ct, match)
			}
		}

		return matches, nil
	}

	var files []string
	// some safetensors files do not properly match "application/octet-stream", so skip checking their contentType
	if st, _ := glob(filepath.Join(path, "model*.safetensors"), ""); len(st) > 0 {
		// safetensors files might be unresolved git lfs references; skip if they are
		// covers model-x-of-y.safetensors, model.fp32-x-of-y.safetensors, model.safetensors
		files = append(files, st...)
	} else if st, _ := glob(filepath.Join(path, "consolidated*.safetensors"), ""); len(st) > 0 {
		// covers consolidated.safetensors
		files = append(files, st...)
	} else if pt, _ := glob(filepath.Join(path, "pytorch_model*.bin"), "application/zip"); len(pt) > 0 {
		// pytorch files might also be unresolved git lfs references; skip if they are
		// covers pytorch_model-x-of-y.bin, pytorch_model.fp32-x-of-y.bin, pytorch_model.bin
		files = append(files, pt...)
	} else if pt, _ := glob(filepath.Join(path, "consolidated*.pth"), "application/zip"); len(pt) > 0 {
		// pytorch files might also be unresolved git lfs references; skip if they are
		// covers consolidated.x.pth, consolidated.pth
		files = append(files, pt...)
	} else if gg, _ := glob(filepath.Join(path, "*.gguf"), "application/octet-stream"); len(gg) > 0 {
		// covers gguf files ending in .gguf
		files = append(files, gg...)
	} else if gg, _ := glob(filepath.Join(path, "*.bin"), "application/octet-stream"); len(gg) > 0 {
		// covers gguf files ending in .bin
		files = append(files, gg...)
	} else {
		return nil, ErrModelNotFound
	}

	// add configuration files, json files are detected as text/plain
	js, err := glob(filepath.Join(path, "*.json"), "text/plain")
	if err != nil {
		return nil, err
	}
	files = append(files, js...)

	// bert models require a nested config.json
	// TODO(mxyng): merge this with the glob above
	js, err = glob(filepath.Join(path, "**/*.json"), "text/plain")
	if err != nil {
		return nil, err
	}
	files = append(files, js...)

	// add tokenizer.model if it exists (tokenizer.json is automatically picked up by the previous glob)
	// tokenizer.model might be a unresolved git lfs reference; error if it is
	if tks, _ := glob(filepath.Join(path, "tokenizer.model"), "application/octet-stream"); len(tks) > 0 {
		files = append(files, tks...)
	} else if tks, _ := glob(filepath.Join(path, "**/tokenizer.model"), "text/plain"); len(tks) > 0 {
		// some times tokenizer.model is in a subdirectory (e.g. meta-llama/Meta-Llama-3-8B)
		files = append(files, tks...)
	}

	return files, nil
}

type Command struct {
	Name string
	Args string
}

func (c Command) String() string {
	var sb strings.Builder
	switch c.Name {
	case "model":
		fmt.Fprintf(&sb, "FROM %s", c.Args)
	case "license", "template", "system", "adapter", "renderer", "parser", "requires":
		fmt.Fprintf(&sb, "%s %s", strings.ToUpper(c.Name), quote(c.Args))
	case "message":
		role, message, _ := strings.Cut(c.Args, ": ")
		fmt.Fprintf(&sb, "MESSAGE %s %s", role, quote(message))
	default:
		fmt.Fprintf(&sb, "PARAMETER %s %s", c.Name, quote(c.Args))
	}

	return sb.String()
}

type state int

const (
	stateNil state = iota
	stateName
	stateValue
	stateParameter
	stateMessage
	stateComment
)

var (
	errMissingFrom        = errors.New("no FROM line")
	errInvalidMessageRole = errors.New("message role must be one of \"system\", \"user\", or \"assistant\"")
	errInvalidCommand     = errors.New("command must be one of \"from\", \"license\", \"template\", \"system\", \"adapter\", \"renderer\", \"parser\", \"parameter\", \"message\", or \"requires\"")
)

type ParserError struct {
	LineNumber int
	Msg        string
}

func (e *ParserError) Error() string {
	if e.LineNumber > 0 {
		return fmt.Sprintf("(line %d): %s", e.LineNumber, e.Msg)
	}
	return e.Msg
}

func ParseFile(r io.Reader) (*Modelfile, error) {
	var cmd Command
	var curr state
	var currLine int = 1
	var b bytes.Buffer
	var role string

	var f Modelfile

	tr := unicode.BOMOverride(unicode.UTF8.NewDecoder())
	br := bufio.NewReader(transform.NewReader(r, tr))

	for {
		r, _, err := br.ReadRune()
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		if isNewline(r) {
			currLine++
		}

		next, r, err := parseRuneForState(r, curr)
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: %s", err, b.String())
		} else if err != nil {
			return nil, &ParserError{
				LineNumber: currLine,
				Msg:        err.Error(),
			}
		}

		// process the state transition, some transitions need to be intercepted and redirected
		if next != curr {
			switch curr {
			case stateName:
				if !isValidCommand(b.String()) {
					return nil, &ParserError{
						LineNumber: currLine,
						Msg:        errInvalidCommand.Error(),
					}
				}

				// next state sometimes depends on the current buffer value
				switch s := strings.ToLower(b.String()); s {
				case "from":
					cmd.Name = "model"
				case "parameter":
					// transition to stateParameter which sets command name
					next = stateParameter
				case "message":
					// transition to stateMessage which validates the message role
					next = stateMessage
					fallthrough
				default:
					cmd.Name = s
				}
			case stateParameter:
				cmd.Name = b.String()
			case stateMessage:
				if !isValidMessageRole(b.String()) {
					return nil, &ParserError{
						LineNumber: currLine,
						Msg:        errInvalidMessageRole.Error(),
					}
				}

				role = b.String()
			case stateComment, stateNil:
				// pass
			case stateValue:
				s, ok := unquote(strings.TrimSpace(b.String()))
				if !ok || isSpace(r) {
					if _, err := b.WriteRune(r); err != nil {
						return nil, err
					}

					continue
				}

				if role != "" {
					s = role + ": " + s
					role = ""
				}

				cmd.Args = s
				f.Commands = append(f.Commands, cmd)
			}

			b.Reset()
			curr = next
		}

		if strconv.IsPrint(r) {
			if _, err := b.WriteRune(r); err != nil {
				return nil, err
			}
		}
	}

	// flush the buffer
	switch curr {
	case stateComment, stateNil:
		// pass; nothing to flush
	case stateValue:
		s, ok := unquote(strings.TrimSpace(b.String()))
		if !ok {
			return nil, io.ErrUnexpectedEOF
		}

		if role != "" {
			s = role + ": " + s
		}

		cmd.Args = s
		f.Commands = append(f.Commands, cmd)
	default:
		return nil, io.ErrUnexpectedEOF
	}

	for _, cmd := range f.Commands {
		if cmd.Name == "model" {
			return &f, nil
		}
	}

	return nil, errMissingFrom
}

func parseRuneForState(r rune, cs state) (state, rune, error) {
	switch cs {
	case stateNil:
		switch {
		case r == '#':
			return stateComment, 0, nil
		case isSpace(r), isNewline(r):
			return stateNil, 0, nil
		default:
			return stateName, r, nil
		}
	case stateName:
		switch {
		case isAlpha(r):
			return stateName, r, nil
		case isSpace(r):
			return stateValue, 0, nil
		default:
			return stateNil, 0, errInvalidCommand
		}
	case stateValue:
		switch {
		case isNewline(r):
			return stateNil, r, nil
		case isSpace(r):
			return stateNil, r, nil
		default:
			return stateValue, r, nil
		}
	case stateParameter:
		switch {
		case isAlpha(r), isNumber(r), r == '_':
			return stateParameter, r, nil
		case isSpace(r):
			return stateValue, 0, nil
		default:
			return stateNil, 0, io.ErrUnexpectedEOF
		}
	case stateMessage:
		switch {
		case isAlpha(r):
			return stateMessage, r, nil
		case isSpace(r):
			return stateValue, 0, nil
		default:
			return stateNil, 0, io.ErrUnexpectedEOF
		}
	case stateComment:
		switch {
		case isNewline(r):
			return stateNil, 0, nil
		default:
			return stateComment, 0, nil
		}
	default:
		return stateNil, 0, errors.New("")
	}
}

func quote(s string) string {
	if strings.Contains(s, "\n") || strings.HasPrefix(s, " ") || strings.HasSuffix(s, " ") {
		if strings.Contains(s, "\"") {
			return `"""` + s + `"""`
		}

		return `"` + s + `"`
	}

	return s
}

func unquote(s string) (string, bool) {
	// TODO: single quotes
	if len(s) >= 3 && s[:3] == `"""` {
		if len(s) >= 6 && s[len(s)-3:] == `"""` {
			return s[3 : len(s)-3], true
		}

		return "", false
	}

	if len(s) >= 1 && s[0] == '"' {
		if len(s) >= 2 && s[len(s)-1] == '"' {
			return s[1 : len(s)-1], true
		}

		return "", false
	}

	return s, true
}

func isAlpha(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'
}

func isNumber(r rune) bool {
	return r >= '0' && r <= '9'
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t'
}

func isNewline(r rune) bool {
	return r == '\r' || r == '\n'
}

func isValidMessageRole(role string) bool {
	return role == "system" || role == "user" || role == "assistant"
}

func isValidCommand(cmd string) bool {
	switch strings.ToLower(cmd) {
	case "from", "license", "template", "system", "adapter", "renderer", "parser", "parameter", "message", "requires":
		return true
	default:
		return false
	}
}

func expandPathImpl(path, relativeDir string, currentUserFunc func() (*user.User, error), lookupUserFunc func(string) (*user.User, error)) (string, error) {
	if filepath.IsAbs(path) || strings.HasPrefix(path, "\\") || strings.HasPrefix(path, "/") {
		return filepath.Abs(path)
	} else if strings.HasPrefix(path, "~") {
		var homeDir string

		if path == "~" || strings.HasPrefix(path, "~/") {
			// Current user's home directory
			currentUser, err := currentUserFunc()
			if err != nil {
				return "", fmt.Errorf("failed to get current user: %w", err)
			}
			homeDir = currentUser.HomeDir
			path = strings.TrimPrefix(path, "~")
		} else {
			// Specific user's home directory
			parts := strings.SplitN(path[1:], "/", 2)
			userInfo, err := lookupUserFunc(parts[0])
			if err != nil {
				return "", fmt.Errorf("failed to find user '%s': %w", parts[0], err)
			}
			homeDir = userInfo.HomeDir
			if len(parts) > 1 {
				path = "/" + parts[1]
			} else {
				path = ""
			}
		}

		path = filepath.Join(homeDir, path)
	} else {
		path = filepath.Join(relativeDir, path)
	}

	return filepath.Abs(path)
}

func expandPath(path, relativeDir string) (string, error) {
	return expandPathImpl(path, relativeDir, user.Current, user.Lookup)
}

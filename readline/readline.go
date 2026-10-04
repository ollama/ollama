package readline

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

type Prompt struct {
	Prompt         string
	AltPrompt      string
	Placeholder    string
	AltPlaceholder string
	UseAlt         bool
}

func (p *Prompt) prompt() string {
	if p.UseAlt {
		return p.AltPrompt
	}
	return p.Prompt
}

func (p *Prompt) placeholder() string {
	if p.UseAlt {
		return p.AltPlaceholder
	}
	return p.Placeholder
}

type Terminal struct {
	reader  *bufio.Reader
	rawmode bool
	termios any
}

type Instance struct {
	Prompt      *Prompt
	Terminal    *Terminal
	History     *History
	Pasting     bool
	Prefill     string
	pastedLines []string
}

func New(prompt Prompt) (*Instance, error) {
	term, err := NewTerminal()
	if err != nil {
		return nil, err
	}

	history, err := NewHistory()
	if err != nil {
		return nil, err
	}

	return &Instance{
		Prompt:   &prompt,
		Terminal: term,
		History:  history,
	}, nil
}

func (i *Instance) Readline() (string, error) {
	if !i.Terminal.rawmode {
		fd := os.Stdin.Fd()
		termios, err := SetRawMode(fd)
		if err != nil {
			return "", err
		}
		i.Terminal.rawmode = true
		i.Terminal.termios = termios
	}

	prompt := i.Prompt.prompt()
	if i.Pasting {
		// force alt prompt when pasting
		prompt = i.Prompt.AltPrompt
	}
	fmt.Print(prompt)

	defer func() {
		fd := os.Stdin.Fd()
		//nolint:errcheck
		UnsetRawMode(fd, i.Terminal.termios)
		i.Terminal.rawmode = false
	}()

	buf, _ := NewBuffer(i.Prompt)

	// Prefill the buffer with any text that we received from an external editor
	if i.Prefill != "" {
		lines := strings.Split(i.Prefill, "\n")
		i.Prefill = ""
		for idx, l := range lines {
			for _, r := range l {
				buf.Add(r)
			}
			if idx < len(lines)-1 {
				i.pastedLines = append(i.pastedLines, buf.String())
				buf.Buf.Clear()
				buf.Pos = 0
				buf.DisplayPos = 0
				buf.LineHasSpace.Clear()
				fmt.Println()
				fmt.Print(i.Prompt.AltPrompt)
				i.Prompt.UseAlt = true
			}
		}
	}

	var esc bool
	var escex bool
	var metaDel bool

	var currentLineBuf []rune

	// draining tracks if we're processing buffered input from cooked mode.
	// In cooked mode Enter sends \n, but in raw mode Ctrl+J sends \n.
	// We treat \n from cooked mode as submit, not multiline.
	// We check Buffered() after the first read since the bufio buffer is
	// empty until then. This is compatible with """ multiline mode in
	// interactive.go since each Readline() call is independent.
	var draining, stopDraining bool

	for {
		// Apply deferred state change from previous iteration
		if stopDraining {
			draining = false
			stopDraining = false
		}

		// don't show placeholder when pasting unless we're in multiline mode
		showPlaceholder := !i.Pasting || i.Prompt.UseAlt
		if buf.IsEmpty() && showPlaceholder {
			ph := i.Prompt.placeholder()
			fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
		}

		r, err := i.Terminal.Read()

		// After reading, check if there's more buffered data. If so, we're
		// processing cooked-mode input. Once buffer empties, the current
		// char is the last buffered one (still drain it), then stop next iteration.
		if i.Terminal.reader.Buffered() > 0 {
			draining = true
		} else if draining {
			stopDraining = true
		}

		if buf.IsEmpty() {
			fmt.Print(ClearToEOL)
		}

		if err != nil {
			return "", io.EOF
		}

		if escex {
			escex = false

			switch r {
			case KeyUp:
				i.historyPrev(buf, &currentLineBuf)
			case KeyDown:
				i.historyNext(buf, &currentLineBuf)
			case KeyLeft:
				buf.MoveLeft()
			case KeyRight:
				buf.MoveRight()
			case CharBracketedPaste:
				var code string
				for range 3 {
					r, err = i.Terminal.Read()
					if err != nil {
						return "", io.EOF
					}

					code += string(r)
				}
				if code == CharBracketedPasteStart {
					i.Pasting = true
				} else if code == CharBracketedPasteEnd {
					i.Pasting = false
				}
			case KeyDel:
				if buf.DisplaySize() > 0 {
					buf.Delete()
				}
				metaDel = true
			case MetaStart:
				buf.MoveToStart()
			case MetaEnd:
				buf.MoveToEnd()
			default:
				// skip any keys we don't know about
				continue
			}
			continue
		} else if esc {
			esc = false

			switch r {
			case 'b':
				buf.MoveLeftWord()
			case 'f':
				buf.MoveRightWord()
			case CharBackspace:
				buf.DeleteWord()
			case CharEscapeEx:
				escex = true
			}
			continue
		}

		switch r {
		case CharNull:
			continue
		case CharEsc:
			esc = true
		case CharInterrupt:
			i.pastedLines = nil
			i.Prompt.UseAlt = false
			return "", ErrInterrupt
		case CharPrev:
			i.historyPrev(buf, &currentLineBuf)
		case CharNext:
			i.historyNext(buf, &currentLineBuf)
		case CharLineStart:
			buf.MoveToStart()
		case CharLineEnd:
			buf.MoveToEnd()
		case CharBackward:
			buf.MoveLeft()
		case CharForward:
			buf.MoveRight()
		case CharBackspace, CharCtrlH:
			if buf.IsEmpty() && len(i.pastedLines) > 0 {
				lastIdx := len(i.pastedLines) - 1
				prevLine := i.pastedLines[lastIdx]
				i.pastedLines = i.pastedLines[:lastIdx]
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + ClearToEOL)
				if len(i.pastedLines) == 0 {
					fmt.Print(i.Prompt.Prompt)
					i.Prompt.UseAlt = false
				} else {
					fmt.Print(i.Prompt.AltPrompt)
				}
				for _, r := range prevLine {
					buf.Add(r)
				}
			} else {
				buf.Remove()
			}
		case CharTab:
			// todo: convert back to real tabs
			for range 8 {
				buf.Add(' ')
			}
		case CharDelete:
			if buf.DisplaySize() > 0 {
				buf.Delete()
			} else {
				return "", io.EOF
			}
		case CharKill:
			buf.DeleteRemaining()
		case CharCtrlU:
			buf.DeleteBefore()
		case CharCtrlL:
			buf.ClearScreen()
		case CharCtrlW:
			buf.DeleteWord()
		case CharBell:
			output := buf.String()
			numPastedLines := len(i.pastedLines)
			if numPastedLines > 0 {
				output = strings.Join(i.pastedLines, "\n") + "\n" + output
				i.pastedLines = nil
			}

			// Move cursor to the last display line of the current buffer
			currLine := buf.DisplayPos / buf.LineWidth
			lastLine := buf.DisplaySize() / buf.LineWidth
			if lastLine > currLine {
				fmt.Print(CursorDownN(lastLine - currLine))
			}

			// Clear all lines from bottom to top: buffer wrapped lines + pasted lines
			for range lastLine + numPastedLines {
				fmt.Print(CursorBOL + ClearToEOL + CursorUp)
			}
			fmt.Print(CursorBOL + ClearToEOL)

			i.Prompt.UseAlt = false
			return output, ErrEditPrompt
		case CharCtrlZ:
			fd := os.Stdin.Fd()
			return handleCharCtrlZ(fd, i.Terminal.termios)
		case CharCtrlJ:
			// If not draining cooked-mode input, treat as multiline
			if !draining {
				i.pastedLines = append(i.pastedLines, buf.String())
				buf.Buf.Clear()
				buf.Pos = 0
				buf.DisplayPos = 0
				buf.LineHasSpace.Clear()
				fmt.Println()
				fmt.Print(i.Prompt.AltPrompt)
				i.Prompt.UseAlt = true
				continue
			}
			// Draining cooked-mode input: treat \n as submit
			fallthrough
		case CharEnter:
			output := buf.String()
			if len(i.pastedLines) > 0 {
				output = strings.Join(i.pastedLines, "\n") + "\n" + output
				i.pastedLines = nil
			}
			if output != "" {
				i.History.Add(output)
			}
			buf.MoveToEnd()
			fmt.Println()
			i.Prompt.UseAlt = false

			return output, nil
		default:
			if metaDel {
				metaDel = false
				continue
			}
			if r >= CharSpace || r == CharEnter || r == CharCtrlJ {
				buf.Add(r)
			}
		}
	}
}

func (i *Instance) HistoryEnable() {
	i.History.Enabled = true
}

func (i *Instance) HistoryDisable() {
	i.History.Enabled = false
}

func (i *Instance) historyPrev(buf *Buffer, currentLineBuf *[]rune) {
	if i.History.Pos > 0 {
		if i.History.Pos == i.History.Size() {
			*currentLineBuf = []rune(buf.String())
		}
		buf.Replace([]rune(i.History.Prev()))
	}
}

func (i *Instance) historyNext(buf *Buffer, currentLineBuf *[]rune) {
	if i.History.Pos < i.History.Size() {
		buf.Replace([]rune(i.History.Next()))
		if i.History.Pos == i.History.Size() {
			buf.Replace(*currentLineBuf)
		}
	}
}

func NewTerminal() (*Terminal, error) {
	fd := os.Stdin.Fd()
	termios, err := SetRawMode(fd)
	if err != nil {
		return nil, err
	}
	if err := UnsetRawMode(fd, termios); err != nil {
		return nil, err
	}

	t := &Terminal{
		reader: bufio.NewReader(os.Stdin),
	}

	return t, nil
}

func (t *Terminal) Read() (rune, error) {
	r, _, err := t.reader.ReadRune()
	if err != nil {
		return 0, err
	}
	return r, nil
}

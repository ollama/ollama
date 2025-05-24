package readline

import (
	"bufio"
	"fmt"
	"io"
	"os"
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
	outchan chan rune
	rawmode bool
	termios any
}

type Instance struct {
	Prompt   *Prompt
	Terminal *Terminal
	History  *History
	Pasting  bool
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

	var esc bool
	var escex bool
	var metaDel bool

	var currentLineBuf []rune

	for {
		// don't show placeholder when pasting unless we're in multiline mode
		showPlaceholder := !i.Pasting || i.Prompt.UseAlt
		if buf.IsEmpty() && showPlaceholder {
			ph := i.Prompt.placeholder()
			fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
		}

		r, err := i.Terminal.Read()

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
			buf.Remove()
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
		case CharCtrlZ:
			fd := os.Stdin.Fd()
			return handleCharCtrlZ(fd, i.Terminal.termios)
		case CharEnter, CharCtrlJ:
			output := buf.String()
			if output != "" {
				i.History.Add(output)
			}
			buf.MoveToEnd()
			fmt.Println()

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

	t := &Terminal{
		outchan: make(chan rune),
		rawmode: true,
		termios: termios,
	}

	go t.ioloop()

	return t, nil
}

func (t *Terminal) ioloop() {
	buf := bufio.NewReader(os.Stdin)

	for {
		r, _, err := buf.ReadRune()
		if err != nil {
			close(t.outchan)
			break
		}
		t.outchan <- r
	}
}

func (t *Terminal) Read() (rune, error) {
	r, ok := <-t.outchan
	if !ok {
		return 0, io.EOF
	}

	return r, nil
}

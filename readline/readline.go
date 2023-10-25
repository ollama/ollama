package readline

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sync"
	"syscall"
)

type Prompt struct {
	Prompt         string
	AltPrompt      string
	Placeholder    string
	AltPlaceholder string
	UseAlt         bool
}

type Terminal struct {
	m       sync.Mutex
	wg      sync.WaitGroup
	outchan chan rune
}

type Instance struct {
	Prompt   *Prompt
	Terminal *Terminal
	History  *History
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
	prompt := i.Prompt.Prompt
	if i.Prompt.UseAlt {
		prompt = i.Prompt.AltPrompt
	}
	fmt.Printf(prompt)

	termios, err := SetRawMode(syscall.Stdin)
	if err != nil {
		return "", err
	}
	defer UnsetRawMode(syscall.Stdin, termios)

	buf, _ := NewBuffer(i.Prompt)

	var esc bool
	var escex bool
	var metaDel bool
	var bracketedPaste bool
	var ignoreEnter bool

	var currentLineBuf []rune

	for {
		if buf.IsEmpty() {
			ph := i.Prompt.Placeholder
			if i.Prompt.UseAlt {
				ph = i.Prompt.AltPlaceholder
			}
			fmt.Printf(ColorGrey + ph + fmt.Sprintf(CursorLeftN, len(ph)) + ColorDefault)
		}

		r := i.Terminal.ReadRune()
		if buf.IsEmpty() {
			fmt.Printf(ClearToEOL)
		}

		if r == 0 { // io.EOF
			break
		}

		if escex {
			escex = false

			switch r {
			case KeyUp:
				if i.History.Pos > 0 {
					if i.History.Pos == i.History.Size() {
						currentLineBuf = []rune(buf.String())
					}
					buf.Replace(i.History.Prev())
				}
			case KeyDown:
				if i.History.Pos < i.History.Size() {
					buf.Replace(i.History.Next())
					if i.History.Pos == i.History.Size() {
						buf.Replace(currentLineBuf)
					}
				}
			case KeyLeft:
				buf.MoveLeft()
			case KeyRight:
				buf.MoveRight()
			case CharBracketedPaste:
				bracketedPaste = true
			case KeyDel:
				if buf.Size() > 0 {
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
			case CharEscapeEx:
				escex = true
			}
			continue
		}

		switch r {
		case CharBracketedPasteStart:
			if bracketedPaste {
				ignoreEnter = true
			}
		case CharEsc:
			esc = true
		case CharInterrupt:
			return "", ErrInterrupt
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
			for cnt := 0; cnt < 8; cnt++ {
				buf.Add(' ')
			}
		case CharDelete:
			if buf.Size() > 0 {
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
		case CharEnter:
			if !ignoreEnter {
				output := buf.String()
				if output != "" {
					i.History.Add([]rune(output))
				}
				buf.MoveToEnd()
				fmt.Println()
				return output, nil
			}
			fallthrough
		default:
			if metaDel {
				metaDel = false
				continue
			}
			if r >= CharSpace || r == CharEnter {
				buf.Add(r)
			}
		}
	}
	return "", nil
}

func (i *Instance) Close() error {
	return i.Terminal.Close()
}

func (i *Instance) HistoryEnable() {
	i.History.Enabled = true
}

func (i *Instance) HistoryDisable() {
	i.History.Enabled = false
}

func NewTerminal() (*Terminal, error) {
	t := &Terminal{
		outchan: make(chan rune),
	}

	go t.ioloop()

	return t, nil
}

func (t *Terminal) ioloop() {
	buf := bufio.NewReader(os.Stdin)

	for {
		r, _, err := buf.ReadRune()
		if err != nil {
			break
		}
		t.outchan <- r
		if r == 0 { // EOF
			break
		}
	}

}

func (t *Terminal) ReadRune() rune {
	r, ok := <-t.outchan
	if !ok {
		return rune(0)
	}
	return r
}

func (t *Terminal) Close() error {
	close(t.outchan)
	return nil
}

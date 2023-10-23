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

	//fmt.Printf(StartBracketedPaste)
	//defer fmt.Printf(EndBracketedPaste)

	var esc bool
	var escex bool
	var bracketedPaste bool
	var ignoreEnter bool

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

		if esc && escex {
			esc = false
			escex = false

			switch r {
			case KeyUp:
				if i.History.Pos > 0 {
					buf.Replace(i.History.Prev())
				}
			case KeyDown:
				if i.History.Pos < i.History.Size() {
					buf.Replace(i.History.Next())
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
				continue
			default:
				fmt.Printf("--- %d ", r)
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
		case CharEscapeEx:
			if esc {
				escex = true
			} else {
				buf.Add(r)
			}
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
			buf.Add(r)
		}
	}
	return "", nil
}

func (i *Instance) Close() error {
	return i.Terminal.Close()
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
	ch, ok := <-t.outchan
	if !ok {
		return rune(0)
	}
	return ch
}

func (t *Terminal) Close() error {
	close(t.outchan)
	return nil
}

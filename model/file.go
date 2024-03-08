package model

import (
	"bufio"
	"io"
	"iter"
	"strings"
)

type Param struct {
	Key   string
	Value string
}

type Message struct {
	Role    string
	Content string
}

type File struct {
	// From is a required pragma that specifies the source of the model,
	// either on disk, or by reference (see blob.ParseRef).
	From string

	// Optional
	Params   []Param
	Template string
	System   string
	Adapter  string
	Messages []Message

	License string
}

type Error struct {
	Pragma  string
	Message string
}

func (e *Error) Error() string {
	return e.Pragma + ": " + e.Message
}

type Pragma struct {
	// The pragma name
	Name string

	// Args contains the user-defined arguments for the pragma. If no
	// arguments were provided, it is nil.
	Args []string
}

func (p Pragma) Arg(i int) string {
	if i >= len(p.Args) {
		return ""
	}
	return p.Args[i]
}

func Pragmas(r io.Reader) iter.Seq2[Pragma, error] {
	return func(yield func(Pragma, error) bool) {
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			line := sc.Text()

			// TODO(bmizerany): set a max num fields/args to
			// prevent mem bloat
			args := strings.Fields(line)
			if len(args) == 0 {
				continue
			}

			p := Pragma{
				Name: strings.ToUpper(args[0]),
			}
			if p.Name == "MESSAGE" {
				// handle special case where message content
				// is space separated on the _rest_ of the
				// line like: `MESSAGE user Is Ontario in
				// Canada?`
				panic("TODO")
			}
			if len(args) > 1 {
				p.Args = args[1:]
			}
			if !yield(p, nil) {
				return
			}
		}
		if sc.Err() != nil {
			yield(Pragma{}, sc.Err())
		}
	}
}

func Decode(r io.Reader) (File, error) {
	var f File
	for p, err := range Pragmas(r) {
		if err != nil {
			return File{}, err
		}
		switch p.Name {
		case "FROM":
			f.From = p.Arg(0)
		case "PARAMETER":
			f.Params = append(f.Params, Param{
				Key:   strings.ToLower(p.Arg(0)),
				Value: p.Arg(1),
			})
		case "TEMPLATE":
			f.Template = p.Arg(0)
		case "SYSTEM":
			f.System = p.Arg(0)
		case "ADAPTER":
			f.Adapter = p.Arg(0)
		case "MESSAGE":
			f.Messages = append(f.Messages, Message{
				Role:    p.Arg(0),
				Content: p.Arg(1),
			})
		case "LICENSE":
			f.License = p.Arg(0)
		}
	}
	return f, nil
}

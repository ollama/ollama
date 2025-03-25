package sample

var DefaultGrammar = map[string]string{
	"unicode": `\x{hex}{2} | \u{hex}{4} | \U{hex}{8}`,
	"null":    `"null"`,
	"object":  `"{" (kv ("," kv)*)? "}"`,
	"array":   `"[" (value ("," value)*)? "]"`,
	"kv":      `string ":" value`,
	"integer": `"0" | [1-9] [0-9]*`,
	"number":  `"-"? integer frac? exp?`,
	"frac":    `"." [0-9]+`,
	"exp":     `("e" | "E") ("+" | "-") [0-9]+`,
	"string":  `"\"" char* "\""`,
	"escape":  `["/" | "b" | "f" | "n" | "r" | "t" | unicode]`,
	"char":    `[^"\\] | escape`,
	"space":   `(" " | "\t" | "\n" | "\r")*`,
	"hex":     `[0-9] | [a-f] | [A-F]`,
	"boolean": `"true" | "false"`,
	"value":   `object | array | string | number | boolean | "null"`,
}

const jsonString = `object | array`

type StateMachine struct {
	states map[rune]State
}

type State struct {
	NextStates []string
	// bitmask?
	Mask       []bool
	IsTerminal bool
}

func NewStateMachine(grammar map[string]string, startRule string) *StateMachine {
	states := make(map[rune]State)

	var cumu string
	flag := false
	for _, r := range startRule {
		if r == '"' {
			flag = !flag
		}
		if flag {
			cumu += string(r)
		}
	}

	sm := &StateMachine{
		states: states,
	}
	return sm
}

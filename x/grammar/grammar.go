//go:build mlx

// Package grammar provides GPU-accelerated constrained decoding using MLX.
// It compiles EBNF grammars to pushdown automata (pda) with precomputed token masks.
// For JSON Schema conversion, see the grammar/schema subpackage.
package grammar

import (
	"encoding/binary"
	"fmt"
	"io"
	"strings"

	"golang.org/x/exp/ebnf"
)

// stackSymbol represents a symbol that can be pushed onto the pda stack.
type stackSymbol int

const (
	stackEmpty stackSymbol = iota
	// Additional stack symbols will be generated per-grammar
)

// state represents a pda state.
type state int

const (
	stateError  state = -1
	stateStart  state = 0
	stateAccept state = 1
	// Additional states will be generated per-grammar
)

// transition represents a pda transition.
// On input matching Pattern, from FromState with stackTop:
//   - Move to ToState
//   - Pop StackPop symbols, push StackPush symbols
type transition struct {
	FromState state
	stackTop  stackSymbol // What must be on stack top (stackEmpty = don't care)
	Pattern   string      // Input pattern to match (token or character class)
	ToState   state
	StackPop  int           // Number of symbols to pop
	StackPush []stackSymbol // Symbols to push (in order, first pushed first)
}

// pda represents a compiled pushdown automaton.
type pda struct {
	States       int                    // Total number of states
	StackSymbols int                    // Total number of stack symbols
	StartState   state                  // Initial state
	AcceptStates map[state]bool         // Set of accepting states
	Transitions  map[state][]transition // Transitions indexed by from-state

	// For token-level matching
	Terminals []string // All terminal symbols (patterns to match)
}

// newPDA creates an empty pda.
func newPDA() *pda {
	return &pda{
		States:       2, // Error and Start
		StackSymbols: 1, // Empty
		StartState:   stateStart,
		AcceptStates: make(map[state]bool),
		Transitions:  make(map[state][]transition),
		Terminals:    make([]string, 0),
	}
}

// addState adds a new state and returns its ID.
func (p *pda) addState() state {
	s := state(p.States)
	p.States++
	return s
}

// addStackSymbol adds a new stack symbol and returns its ID.
func (p *pda) addStackSymbol() stackSymbol {
	s := stackSymbol(p.StackSymbols)
	p.StackSymbols++
	return s
}

// addTransition adds a transition to the pda.
func (p *pda) addTransition(t transition) {
	p.Transitions[t.FromState] = append(p.Transitions[t.FromState], t)
}

// addTerminal registers a terminal pattern and returns its index.
func (p *pda) addTerminal(pattern string) int {
	for i, t := range p.Terminals {
		if t == pattern {
			return i
		}
	}
	p.Terminals = append(p.Terminals, pattern)
	return len(p.Terminals) - 1
}

// compiler compiles EBNF grammars to PDAs.
type compiler struct {
	grammar ebnf.Grammar
	pda     *pda

	// Maps production names to their entry/exit states
	prodEntry map[string]state
	prodExit  map[string]state
}

// compile parses an EBNF grammar and compiles it to a pda.
func compile(name string, src io.Reader, start string) (*pda, error) {
	grammar, err := ebnf.Parse(name, src)
	if err != nil {
		return nil, fmt.Errorf("parse grammar: %w", err)
	}

	if err := ebnf.Verify(grammar, start); err != nil {
		return nil, fmt.Errorf("verify grammar: %w", err)
	}

	c := &compiler{
		grammar:   grammar,
		pda:       newPDA(),
		prodEntry: make(map[string]state),
		prodExit:  make(map[string]state),
	}

	// Create entry/exit states for each production
	for name := range grammar {
		c.prodEntry[name] = c.pda.addState()
		c.prodExit[name] = c.pda.addState()
	}

	// compile each production
	for name, prod := range grammar {
		if err := c.compileProduction(name, prod); err != nil {
			return nil, fmt.Errorf("compile production %q: %w", name, err)
		}
	}

	// Set start state to entry of start production
	if entry, ok := c.prodEntry[start]; ok {
		// Add epsilon transition from pda start to grammar start
		c.pda.addTransition(transition{
			FromState: stateStart,
			Pattern:   "", // epsilon
			ToState:   entry,
		})
	} else {
		return nil, fmt.Errorf("start production %q not found", start)
	}

	// Mark exit of start production as accepting
	if exit, ok := c.prodExit[start]; ok {
		c.pda.AcceptStates[exit] = true
	}

	return c.pda, nil
}

// compileString is a convenience function to compile from a string.
func compileString(grammar string, start string) (*pda, error) {
	return compile("grammar", strings.NewReader(grammar), start)
}

func (c *compiler) compileProduction(name string, prod *ebnf.Production) error {
	entry := c.prodEntry[name]
	exit := c.prodExit[name]

	return c.compileExpr(prod.Expr, entry, exit)
}

func (c *compiler) compileExpr(expr ebnf.Expression, entry, exit state) error {
	switch e := expr.(type) {
	case *ebnf.Name:
		return c.compileName(e, entry, exit)
	case *ebnf.Token:
		return c.compileToken(e, entry, exit)
	case ebnf.Sequence:
		return c.compileSequence(e, entry, exit)
	case ebnf.Alternative:
		return c.compileAlternative(e, entry, exit)
	case *ebnf.Option:
		return c.compileOption(e, entry, exit)
	case *ebnf.Repetition:
		return c.compileRepetition(e, entry, exit)
	case *ebnf.Group:
		return c.compileExpr(e.Body, entry, exit)
	case *ebnf.Range:
		return c.compileRange(e, entry, exit)
	case nil:
		// Empty production - direct epsilon transition
		c.pda.addTransition(transition{
			FromState: entry,
			Pattern:   "",
			ToState:   exit,
		})
		return nil
	default:
		return fmt.Errorf("unsupported expression type: %T", expr)
	}
}

func (c *compiler) compileName(n *ebnf.Name, entry, exit state) error {
	// Reference to another production
	prodName := n.String

	prodEntry, ok := c.prodEntry[prodName]
	if !ok {
		return fmt.Errorf("undefined production: %s", prodName)
	}
	prodExit := c.prodExit[prodName]
	// Use a unique stack symbol per call site so returns are unambiguous.
	stackSym := c.pda.addStackSymbol()

	// Push return address, go to production entry
	c.pda.addTransition(transition{
		FromState: entry,
		Pattern:   "", // epsilon
		ToState:   prodEntry,
		StackPush: []stackSymbol{stackSym},
	})

	// On production exit, pop and return
	c.pda.addTransition(transition{
		FromState: prodExit,
		stackTop:  stackSym,
		Pattern:   "", // epsilon
		ToState:   exit,
		StackPop:  1,
	})

	return nil
}

func (c *compiler) compileToken(t *ebnf.Token, entry, exit state) error {
	// terminal symbol - add transition that consumes this token
	pattern := t.String
	c.pda.addTerminal(pattern)

	c.pda.addTransition(transition{
		FromState: entry,
		Pattern:   pattern,
		ToState:   exit,
	})

	return nil
}

func (c *compiler) compileSequence(seq ebnf.Sequence, entry, exit state) error {
	if len(seq) == 0 {
		// Empty sequence - epsilon transition
		c.pda.addTransition(transition{
			FromState: entry,
			Pattern:   "",
			ToState:   exit,
		})
		return nil
	}

	// Chain: entry -> s1 -> s2 -> ... -> exit
	current := entry
	for i, expr := range seq {
		var next state
		if i == len(seq)-1 {
			next = exit
		} else {
			next = c.pda.addState()
		}

		if err := c.compileExpr(expr, current, next); err != nil {
			return err
		}
		current = next
	}

	return nil
}

func (c *compiler) compileAlternative(alt ebnf.Alternative, entry, exit state) error {
	// Each alternative goes from entry to exit
	for _, expr := range alt {
		if err := c.compileExpr(expr, entry, exit); err != nil {
			return err
		}
	}
	return nil
}

func (c *compiler) compileOption(opt *ebnf.Option, entry, exit state) error {
	// Optional: can skip (epsilon) or take the body

	// Epsilon transition (skip)
	c.pda.addTransition(transition{
		FromState: entry,
		Pattern:   "",
		ToState:   exit,
	})

	// Or take the body
	return c.compileExpr(opt.Body, entry, exit)
}

func (c *compiler) compileRepetition(rep *ebnf.Repetition, entry, exit state) error {
	// Repetition {body}: zero or more
	// entry -> exit (skip)
	// entry -> body -> entry (loop back)

	// Skip transition
	c.pda.addTransition(transition{
		FromState: entry,
		Pattern:   "",
		ToState:   exit,
	})

	// Loop: entry -> (body) -> entry
	return c.compileExpr(rep.Body, entry, entry)
}

func (c *compiler) compileRange(r *ebnf.Range, entry, exit state) error {
	// Character range like "a" … "z" or "\u03b1" … "\u03c9"
	begin := strings.Trim(r.Begin.String, "\"")
	end := strings.Trim(r.End.String, "\"")

	// Unescape bounds first (so "\u03b1" works)
	beginUnesc, err := unescapeLiteral(begin)
	if err != nil {
		return fmt.Errorf("invalid range begin: %w", err)
	}
	endUnesc, err := unescapeLiteral(end)
	if err != nil {
		return fmt.Errorf("invalid range end: %w", err)
	}

	// Validate as single runes (not bytes) for Unicode support
	beginRunes := []rune(beginUnesc)
	endRunes := []rune(endUnesc)
	if len(beginRunes) != 1 || len(endRunes) != 1 {
		return fmt.Errorf("range bounds must be single characters: %q..%q", r.Begin.String, r.End.String)
	}

	// Use unescaped rune strings in pattern (consistent with matcher)
	pattern := fmt.Sprintf("[%s-%s]", string(beginRunes[0]), string(endRunes[0]))
	c.pda.addTerminal(pattern)

	c.pda.addTransition(transition{
		FromState: entry,
		Pattern:   pattern,
		ToState:   exit,
	})

	return nil
}

// runtime represents a pda execution instance.
type runtime struct {
	pda   *pda
	state state
	stack []stackSymbol
}

// newRuntime creates a new pda runtime.
func newRuntime(pda *pda) *runtime {
	return &runtime{
		pda:   pda,
		state: pda.StartState,
		stack: make([]stackSymbol, 0, 32),
	}
}

// stackTop returns the top of the stack, or stackEmpty if empty.
func (r *runtime) stackTop() stackSymbol {
	if len(r.stack) == 0 {
		return stackEmpty
	}
	return r.stack[len(r.stack)-1]
}

// isAccepting returns true if we can reach an accepting state via epsilon transitions
// with an empty stack.
func (r *runtime) isAccepting() bool {
	return r.canReachAccept(r.state, r.stack, make(map[stateStackKey]bool))
}

func (r *runtime) canReachAccept(state state, stack []stackSymbol, visited map[stateStackKey]bool) bool {
	// Check if this state is accepting with empty stack
	if r.pda.AcceptStates[state] && len(stack) == 0 {
		return true
	}

	// Avoid infinite loops
	key := stateStackKey{state: state, stackSig: stackSignature(stack)}
	if visited[key] {
		return false
	}
	visited[key] = true

	// Try epsilon transitions
	for _, t := range r.pda.Transitions[state] {
		if t.Pattern != "" {
			continue // Not epsilon
		}

		// Check stack constraint
		stackTop := stackEmpty
		if len(stack) > 0 {
			stackTop = stack[len(stack)-1]
		}
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}

		// Simulate stack operations
		newStack := make([]stackSymbol, len(stack))
		copy(newStack, stack)

		if t.StackPop > 0 && len(newStack) >= t.StackPop {
			newStack = newStack[:len(newStack)-t.StackPop]
		}
		newStack = append(newStack, t.StackPush...)

		if r.canReachAccept(t.ToState, newStack, visited) {
			return true
		}
	}

	return false
}

// Reset resets the runtime to initial state.
func (r *runtime) Reset() {
	r.state = r.pda.StartState
	r.stack = r.stack[:0]
}

// validInputs returns all valid input patterns from current state.
func (r *runtime) validInputs() []string {
	var valid []string
	seen := make(map[string]bool)
	visited := make(map[stateStackKey]bool)

	// Make a copy of the stack for simulation
	simStack := make([]stackSymbol, len(r.stack))
	copy(simStack, r.stack)

	r.collectValidInputs(r.state, simStack, seen, visited, &valid)
	return valid
}

// stateStackKey is used to detect cycles in epsilon closure
type stateStackKey struct {
	state    state
	stackSig string
}

func stackSignature(stack []stackSymbol) string {
	if len(stack) == 0 {
		return ""
	}
	buf := make([]byte, len(stack)*8)
	for i, sym := range stack {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(sym))
	}
	return string(buf)
}

func (r *runtime) collectValidInputs(state state, simStack []stackSymbol, seen map[string]bool, visited map[stateStackKey]bool, valid *[]string) {
	// Get stack top for comparisons
	stackTop := stackEmpty
	if len(simStack) > 0 {
		stackTop = simStack[len(simStack)-1]
	}

	// Check for cycles to avoid infinite loops
	key := stateStackKey{state: state, stackSig: stackSignature(simStack)}
	if visited[key] {
		return
	}
	visited[key] = true

	transitions := r.pda.Transitions[state]

	for _, t := range transitions {
		// Check stack constraint
		if t.stackTop != stackEmpty && t.stackTop != stackTop {
			continue
		}

		if t.Pattern == "" {
			// Epsilon transition - simulate stack operations
			newStack := make([]stackSymbol, len(simStack))
			copy(newStack, simStack)

			// Pop
			if t.StackPop > 0 {
				if len(newStack) < t.StackPop {
					continue // Can't pop, skip this transition
				}
				newStack = newStack[:len(newStack)-t.StackPop]
			}

			// Push
			newStack = append(newStack, t.StackPush...)

			r.collectValidInputs(t.ToState, newStack, seen, visited, valid)
		} else {
			// terminal - add if not seen
			if !seen[t.Pattern] {
				seen[t.Pattern] = true
				*valid = append(*valid, t.Pattern)
			}
		}
	}
}

// matchesPattern checks if input matches a pattern.
// Patterns can be:
// - Exact strings: "a", "{", "true"
// - Character ranges: "[a-z]", "[0-9]", "[#-~]"
func matchesPattern(input, pattern string) bool {
	// Exact match
	if input == pattern {
		return true
	}

	// Check for character range pattern [X-Y]
	if len(pattern) == 5 && pattern[0] == '[' && pattern[2] == '-' && pattern[4] == ']' {
		if len(input) != 1 {
			return false
		}
		ch := input[0]
		low := pattern[1]
		high := pattern[3]
		return ch >= low && ch <= high
	}

	return false
}

// Accept tries to accept an input, returning true if successful.
func (r *runtime) Accept(input string) bool {
	return r.accept(input, make(map[stateStackKey]bool))
}

func (r *runtime) accept(input string, visited map[stateStackKey]bool) bool {
	key := stateStackKey{state: r.state, stackSig: stackSignature(r.stack)}
	if visited[key] {
		return false
	}
	visited[key] = true

	transitions := r.pda.Transitions[r.state]

	// First, process any epsilon transitions to reach a state that can accept input
	// This is a simplified version - full implementation would need epsilon closure
	for _, t := range transitions {
		if matchesPattern(input, t.Pattern) {
			if t.stackTop != stackEmpty && t.stackTop != r.stackTop() {
				continue
			}
			if t.StackPop > len(r.stack) {
				continue
			}

			// Apply transition
			r.applyTransition(t)
			return true
		}
	}

	// Try epsilon transitions first
	for _, t := range transitions {
		if t.Pattern == "" {
			if t.stackTop != stackEmpty && t.stackTop != r.stackTop() {
				continue
			}
			if t.StackPop > len(r.stack) {
				continue
			}

			// Save state for backtracking
			oldState := r.state
			oldStack := make([]stackSymbol, len(r.stack))
			copy(oldStack, r.stack)

			r.applyTransition(t)

			if r.accept(input, visited) {
				return true
			}

			// Backtrack
			r.state = oldState
			r.stack = oldStack
		}
	}

	return false
}

func (r *runtime) applyTransition(t transition) {
	// Pop
	if t.StackPop > 0 && len(r.stack) >= t.StackPop {
		r.stack = r.stack[:len(r.stack)-t.StackPop]
	}

	// Push
	r.stack = append(r.stack, t.StackPush...)

	// Move to new state
	r.state = t.ToState
}

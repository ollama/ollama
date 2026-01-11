//go:build mlx

// Package constrained provides GPU-accelerated constrained decoding using MLX.
// It compiles EBNF grammars to pushdown automata (PDA) with precomputed token masks.
package constrained

import (
	"fmt"
	"io"
	"strings"

	"golang.org/x/exp/ebnf"
)

// StackSymbol represents a symbol that can be pushed onto the PDA stack.
type StackSymbol int

const (
	StackEmpty StackSymbol = iota
	// Additional stack symbols will be generated per-grammar
)

// State represents a PDA state.
type State int

const (
	StateError  State = -1
	StateStart  State = 0
	StateAccept State = 1
	// Additional states will be generated per-grammar
)

// Transition represents a PDA transition.
// On input matching Pattern, from FromState with StackTop:
//   - Move to ToState
//   - Pop StackPop symbols, push StackPush symbols
type Transition struct {
	FromState State
	StackTop  StackSymbol // What must be on stack top (StackEmpty = don't care)
	Pattern   string      // Input pattern to match (token or character class)
	ToState   State
	StackPop  int           // Number of symbols to pop
	StackPush []StackSymbol // Symbols to push (in order, first pushed first)
}

// PDA represents a compiled pushdown automaton.
type PDA struct {
	States       int                    // Total number of states
	StackSymbols int                    // Total number of stack symbols
	StartState   State                  // Initial state
	AcceptStates map[State]bool         // Set of accepting states
	Transitions  map[State][]Transition // Transitions indexed by from-state

	// For token-level matching
	Terminals []string // All terminal symbols (patterns to match)
}

// NewPDA creates an empty PDA.
func NewPDA() *PDA {
	return &PDA{
		States:       2, // Error and Start
		StackSymbols: 1, // Empty
		StartState:   StateStart,
		AcceptStates: make(map[State]bool),
		Transitions:  make(map[State][]Transition),
		Terminals:    make([]string, 0),
	}
}

// AddState adds a new state and returns its ID.
func (p *PDA) AddState() State {
	s := State(p.States)
	p.States++
	return s
}

// AddStackSymbol adds a new stack symbol and returns its ID.
func (p *PDA) AddStackSymbol() StackSymbol {
	s := StackSymbol(p.StackSymbols)
	p.StackSymbols++
	return s
}

// AddTransition adds a transition to the PDA.
func (p *PDA) AddTransition(t Transition) {
	p.Transitions[t.FromState] = append(p.Transitions[t.FromState], t)
}

// AddTerminal registers a terminal pattern and returns its index.
func (p *PDA) AddTerminal(pattern string) int {
	for i, t := range p.Terminals {
		if t == pattern {
			return i
		}
	}
	p.Terminals = append(p.Terminals, pattern)
	return len(p.Terminals) - 1
}

// Compiler compiles EBNF grammars to PDAs.
type Compiler struct {
	grammar ebnf.Grammar
	pda     *PDA

	// Maps production names to their entry/exit states
	prodEntry map[string]State
	prodExit  map[string]State

	// Stack symbols for each production (for recursion)
	prodStack map[string]StackSymbol
}

// Compile parses an EBNF grammar and compiles it to a PDA.
func Compile(name string, src io.Reader, start string) (*PDA, error) {
	grammar, err := ebnf.Parse(name, src)
	if err != nil {
		return nil, fmt.Errorf("parse grammar: %w", err)
	}

	if err := ebnf.Verify(grammar, start); err != nil {
		return nil, fmt.Errorf("verify grammar: %w", err)
	}

	c := &Compiler{
		grammar:   grammar,
		pda:       NewPDA(),
		prodEntry: make(map[string]State),
		prodExit:  make(map[string]State),
		prodStack: make(map[string]StackSymbol),
	}

	// Create entry/exit states for each production
	for name := range grammar {
		c.prodEntry[name] = c.pda.AddState()
		c.prodExit[name] = c.pda.AddState()
		c.prodStack[name] = c.pda.AddStackSymbol()
	}

	// Compile each production
	for name, prod := range grammar {
		if err := c.compileProduction(name, prod); err != nil {
			return nil, fmt.Errorf("compile production %q: %w", name, err)
		}
	}

	// Set start state to entry of start production
	if entry, ok := c.prodEntry[start]; ok {
		// Add epsilon transition from PDA start to grammar start
		c.pda.AddTransition(Transition{
			FromState: StateStart,
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

// CompileString is a convenience function to compile from a string.
func CompileString(grammar string, start string) (*PDA, error) {
	return Compile("grammar", strings.NewReader(grammar), start)
}

func (c *Compiler) compileProduction(name string, prod *ebnf.Production) error {
	entry := c.prodEntry[name]
	exit := c.prodExit[name]

	return c.compileExpr(prod.Expr, entry, exit)
}

func (c *Compiler) compileExpr(expr ebnf.Expression, entry, exit State) error {
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
		c.pda.AddTransition(Transition{
			FromState: entry,
			Pattern:   "",
			ToState:   exit,
		})
		return nil
	default:
		return fmt.Errorf("unsupported expression type: %T", expr)
	}
}

func (c *Compiler) compileName(n *ebnf.Name, entry, exit State) error {
	// Reference to another production
	prodName := n.String

	prodEntry, ok := c.prodEntry[prodName]
	if !ok {
		return fmt.Errorf("undefined production: %s", prodName)
	}
	prodExit := c.prodExit[prodName]
	stackSym := c.prodStack[prodName]

	// Push return address, go to production entry
	c.pda.AddTransition(Transition{
		FromState: entry,
		Pattern:   "", // epsilon
		ToState:   prodEntry,
		StackPush: []StackSymbol{stackSym},
	})

	// On production exit, pop and return
	c.pda.AddTransition(Transition{
		FromState: prodExit,
		StackTop:  stackSym,
		Pattern:   "", // epsilon
		ToState:   exit,
		StackPop:  1,
	})

	return nil
}

func (c *Compiler) compileToken(t *ebnf.Token, entry, exit State) error {
	// Terminal symbol - add transition that consumes this token
	pattern := t.String
	c.pda.AddTerminal(pattern)

	c.pda.AddTransition(Transition{
		FromState: entry,
		Pattern:   pattern,
		ToState:   exit,
	})

	return nil
}

func (c *Compiler) compileSequence(seq ebnf.Sequence, entry, exit State) error {
	if len(seq) == 0 {
		// Empty sequence - epsilon transition
		c.pda.AddTransition(Transition{
			FromState: entry,
			Pattern:   "",
			ToState:   exit,
		})
		return nil
	}

	// Chain: entry -> s1 -> s2 -> ... -> exit
	current := entry
	for i, expr := range seq {
		var next State
		if i == len(seq)-1 {
			next = exit
		} else {
			next = c.pda.AddState()
		}

		if err := c.compileExpr(expr, current, next); err != nil {
			return err
		}
		current = next
	}

	return nil
}

func (c *Compiler) compileAlternative(alt ebnf.Alternative, entry, exit State) error {
	// Each alternative goes from entry to exit
	for _, expr := range alt {
		if err := c.compileExpr(expr, entry, exit); err != nil {
			return err
		}
	}
	return nil
}

func (c *Compiler) compileOption(opt *ebnf.Option, entry, exit State) error {
	// Optional: can skip (epsilon) or take the body

	// Epsilon transition (skip)
	c.pda.AddTransition(Transition{
		FromState: entry,
		Pattern:   "",
		ToState:   exit,
	})

	// Or take the body
	return c.compileExpr(opt.Body, entry, exit)
}

func (c *Compiler) compileRepetition(rep *ebnf.Repetition, entry, exit State) error {
	// Repetition {body}: zero or more
	// entry -> exit (skip)
	// entry -> body -> entry (loop back)

	// Skip transition
	c.pda.AddTransition(Transition{
		FromState: entry,
		Pattern:   "",
		ToState:   exit,
	})

	// Loop: entry -> (body) -> entry
	return c.compileExpr(rep.Body, entry, entry)
}

func (c *Compiler) compileRange(r *ebnf.Range, entry, exit State) error {
	// Character range like "a" … "z"
	// For now, create a pattern that represents this range
	pattern := fmt.Sprintf("[%s-%s]", r.Begin.String, r.End.String)
	c.pda.AddTerminal(pattern)

	c.pda.AddTransition(Transition{
		FromState: entry,
		Pattern:   pattern,
		ToState:   exit,
	})

	return nil
}

// Runtime represents a PDA execution instance.
type Runtime struct {
	pda   *PDA
	state State
	stack []StackSymbol
}

// NewRuntime creates a new PDA runtime.
func NewRuntime(pda *PDA) *Runtime {
	return &Runtime{
		pda:   pda,
		state: pda.StartState,
		stack: make([]StackSymbol, 0, 32),
	}
}

// State returns the current state.
func (r *Runtime) State() State {
	return r.state
}

// StackTop returns the top of the stack, or StackEmpty if empty.
func (r *Runtime) StackTop() StackSymbol {
	if len(r.stack) == 0 {
		return StackEmpty
	}
	return r.stack[len(r.stack)-1]
}

// IsAccepting returns true if we can reach an accepting state via epsilon transitions
// with an empty stack.
func (r *Runtime) IsAccepting() bool {
	return r.canReachAccept(r.state, r.stack, make(map[State]bool))
}

func (r *Runtime) canReachAccept(state State, stack []StackSymbol, visited map[State]bool) bool {
	// Check if this state is accepting with empty stack
	if r.pda.AcceptStates[state] && len(stack) == 0 {
		return true
	}

	// Avoid infinite loops
	if visited[state] {
		return false
	}
	visited[state] = true

	// Try epsilon transitions
	for _, t := range r.pda.Transitions[state] {
		if t.Pattern != "" {
			continue // Not epsilon
		}

		// Check stack constraint
		stackTop := StackEmpty
		if len(stack) > 0 {
			stackTop = stack[len(stack)-1]
		}
		if t.StackTop != StackEmpty && t.StackTop != stackTop {
			continue
		}

		// Simulate stack operations
		newStack := make([]StackSymbol, len(stack))
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
func (r *Runtime) Reset() {
	r.state = r.pda.StartState
	r.stack = r.stack[:0]
}

// ValidInputs returns all valid input patterns from current state.
func (r *Runtime) ValidInputs() []string {
	var valid []string
	seen := make(map[string]bool)

	r.collectValidInputs(r.state, r.StackTop(), seen, &valid)
	return valid
}

func (r *Runtime) collectValidInputs(state State, stackTop StackSymbol, seen map[string]bool, valid *[]string) {
	transitions := r.pda.Transitions[state]

	for _, t := range transitions {
		// Check stack constraint
		if t.StackTop != StackEmpty && t.StackTop != stackTop {
			continue
		}

		if t.Pattern == "" {
			// Epsilon transition - follow it
			newStackTop := stackTop
			if t.StackPop > 0 && len(r.stack) > 0 {
				// Would pop - simulate new stack top
				if len(r.stack) > t.StackPop {
					newStackTop = r.stack[len(r.stack)-t.StackPop-1]
				} else {
					newStackTop = StackEmpty
				}
			}
			if len(t.StackPush) > 0 {
				newStackTop = t.StackPush[len(t.StackPush)-1]
			}
			r.collectValidInputs(t.ToState, newStackTop, seen, valid)
		} else {
			// Terminal - add if not seen
			if !seen[t.Pattern] {
				seen[t.Pattern] = true
				*valid = append(*valid, t.Pattern)
			}
		}
	}
}

// Accept tries to accept an input, returning true if successful.
func (r *Runtime) Accept(input string) bool {
	transitions := r.pda.Transitions[r.state]

	// First, process any epsilon transitions to reach a state that can accept input
	// This is a simplified version - full implementation would need epsilon closure
	for _, t := range transitions {
		if t.Pattern == input {
			if t.StackTop != StackEmpty && t.StackTop != r.StackTop() {
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
			if t.StackTop != StackEmpty && t.StackTop != r.StackTop() {
				continue
			}

			// Save state for backtracking
			oldState := r.state
			oldStack := make([]StackSymbol, len(r.stack))
			copy(oldStack, r.stack)

			r.applyTransition(t)

			if r.Accept(input) {
				return true
			}

			// Backtrack
			r.state = oldState
			r.stack = oldStack
		}
	}

	return false
}

func (r *Runtime) applyTransition(t Transition) {
	// Pop
	if t.StackPop > 0 && len(r.stack) >= t.StackPop {
		r.stack = r.stack[:len(r.stack)-t.StackPop]
	}

	// Push
	r.stack = append(r.stack, t.StackPush...)

	// Move to new state
	r.state = t.ToState
}

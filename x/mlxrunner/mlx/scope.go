package mlx

import (
	"fmt"
)

// Array lifetimes
//
// Every handle belongs to a scope, a set of arrays freed together. A
// function scope is entered with Scoped, ScopedArrays, ScopedEval, or
// ScopedAsyncEval and ends when the function returns; a held scope is
// created with NewScope and ends when its holder closes it. A function
// scope frees what was created in it or detached into it; a held scope
// frees what was attached to it. An array moves between scopes in three
// ways only: by being returned from a function scope to the caller's scope,
// by Attach into a held scope, or by Detach from a held scope back into the
// current one for a caller that still reads them.
//
// MLX frees a buffer once no handle and no queued graph refers to it, so a
// graph is built in one function scope and evaluated after that scope ends,
// and the eval frees each intermediate as it consumes it. The function that
// finishes a graph opens the scope, returns the arrays that leave it, and
// discards or closes inside it whatever the graph consumed. An operation
// that only adds to its caller's graph, a model forward, a cache update, a
// sampling distribution, builds into the open scope and never opens its
// own; one that finishes a graph of its own, a cache copy, a sample, opens
// one like any other finisher. Whoever needs the values evaluates them once
// the scope has ended: the finisher itself with ScopedEval or
// ScopedAsyncEval, or its caller with Eval or AsyncEval. A holder holds
// what outlives the function that produced it, and only a holder gives its
// arrays up.

type Scope struct {
	arrays []*Array
	parent *Scope
	// noEscape refuses to let an array move out of the scope.
	noEscape bool
}

// Function scopes

// Scoped runs fn in a function scope. Arrays created or released inside it
// are freed when fn returns.
func Scoped(fn func()) {
	s := enterScope()
	defer exitScope(s)
	fn()
}

// ScopedArrays runs fn in a function scope and moves the arrays it returns to
// the caller's scope.
func ScopedArrays(fn func() []*Array) []*Array {
	s := enterScope()
	defer exitScope(s)
	ts := fn()
	escape(ts...)
	return ts
}

// ScopedEval runs fn in a function scope, moves the arrays it returns to the
// caller's scope, ends the scope, and then evaluates them.
func ScopedEval(fn func() []*Array) []*Array {
	ts := ScopedArrays(fn)
	Eval(ts...)
	return ts
}

// ScopedAsyncEval is ScopedEval with an asynchronous evaluation.
func ScopedAsyncEval(fn func() []*Array) []*Array {
	ts := ScopedArrays(fn)
	AsyncEval(ts...)
	return ts
}

// Held scopes

func NewScope() *Scope {
	return &Scope{}
}

// Attach takes arrays from the function scope that built them or from the
// root. An array some held scope already holds, this one included, is that
// holder's to discard or detach first: a second Attach means two owners.
func (s *Scope) Attach(arrays ...*Array) {
	for _, t := range arrays {
		if t == nil {
			continue
		}
		if t.scope != rootScope && t.scope.parent == nil {
			panic(fmt.Sprintf("mlx: array %q is already held", t.name))
		}
		s.take(t)
	}
}

// Detach moves arrays back to the current scope, for a caller that still
// reads them.
func (s *Scope) Detach(arrays ...*Array) {
	for _, t := range arrays {
		if t == nil {
			continue
		}
		if t.scope != s {
			panic(fmt.Sprintf("mlx: array %q is not held by this scope", t.name))
		}
		currentScope.take(t)
	}
}

func (s *Scope) Discard(arrays ...*Array) {
	for _, t := range arrays {
		if t == nil {
			continue
		}
		if t.scope != s {
			panic(fmt.Sprintf("mlx: array %q is not held by this scope", t.name))
		}
		s.remove(t)
		t.free()
	}
}

// Close frees whatever the scope still holds. A nil *Scope holds nothing.
func (s *Scope) Close() {
	if s == nil {
		return
	}
	s.end()
}

// Internals

var (
	rootScope    = &Scope{}
	currentScope = rootScope
)

func enterScope() *Scope {
	s := &Scope{parent: currentScope}
	currentScope = s
	return s
}

// exitScope ends s.
func exitScope(s *Scope) {
	if currentScope != s {
		panic("mlx: scope exited out of order")
	}
	currentScope = s.parent
	s.end()
}

// escape moves arrays of the current scope to the caller's scope so they
// outlive the current one. Arrays held elsewhere are left where they are.
func escape(arrays ...*Array) {
	if currentScope.parent == nil {
		return
	}
	for _, t := range arrays {
		if t != nil && t.scope == currentScope {
			currentScope.parent.take(t)
		}
	}
}

// end frees the arrays in s.
func (s *Scope) end() {
	for _, t := range s.arrays {
		t.free()
	}
	s.arrays = nil
}

// take moves t into s, out of the scope it was in.
func (s *Scope) take(t *Array) {
	if from := t.scope; from != nil {
		if from == s {
			return
		}
		if !t.valid() {
			panic(fmt.Sprintf("mlx: array %q used after its scope ended", t.name))
		}
		if from.noEscape {
			panic(fmt.Sprintf("mlx: array %q escaped a scope that allows no escape", t.name))
		}
		from.remove(t)
	}
	t.scope = s
	s.arrays = append(s.arrays, t)
}

// remove drops t from s's list. An array usually leaves the scope that just
// built it, so the search runs from the end; order in the list is free.
func (s *Scope) remove(t *Array) {
	for i := len(s.arrays) - 1; i >= 0; i-- {
		if s.arrays[i] == t {
			last := len(s.arrays) - 1
			s.arrays[i] = s.arrays[last]
			s.arrays = s.arrays[:last]
			return
		}
	}
}

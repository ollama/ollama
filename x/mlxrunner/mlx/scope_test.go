package mlx

import (
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthreadtest"
)

// A function scope frees what was created in it. What fn returns moves to the
// caller's scope instead, and a returned array the scope does not own stays
// where it is.
func TestScopeFreesWhatIsNotReturned(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		held := NewScope()
		defer held.Close()
		var kept, returned, dropped *Array
		Scoped(func() {
			kept = FromValue(1)
			held.Attach(kept)
			out := ScopedArrays(func() []*Array {
				dropped = FromValue(2)
				return []*Array{FromValue(3), nil, kept}
			})
			returned = out[0]
			if !returned.Valid() {
				t.Fatal("returned array was freed with the scope that created it")
			}
			if dropped.Valid() {
				t.Fatal("array not returned survived its scope")
			}
		})
		if returned.Valid() {
			t.Fatal("returned array survived the scope it was returned into")
		}
		if !kept.Valid() {
			t.Fatal("returning a held array moved it out of its scope")
		}
	})
}

// ScopedEval ends the build scope before it evaluates, so the intermediates
// are gone by then and the returned arrays come back evaluated.
func TestScopedEvalEvaluatesAfterBuild(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		Scoped(func() {
			var tmp *Array
			out := ScopedEval(func() []*Array {
				tmp = FromValue(2)
				return []*Array{FromValue(1).Add(tmp)}
			})
			if tmp.Valid() {
				t.Fatal("intermediate survived the build scope")
			}
			if !out[0].Valid() || out[0].Int() != 3 {
				t.Fatal("returned array was not evaluated after the build scope")
			}
		})
	})
}

// A held scope keeps arrays past the function scope that created them.
// Discard frees one now, Detach hands one to the current scope, and Close
// frees what remains. A scope refuses to discard or detach an array it does
// not hold, and to attach one that is already held, by itself or another
// scope.
func TestHeldScope(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		held, other := NewScope(), NewScope()
		defer other.Close()
		var kept, discarded, detached *Array
		Scoped(func() {
			kept, discarded, detached = FromValue(1), FromValue(2), FromValue(3)
			held.Attach(kept, discarded, detached)
			held.Discard(discarded)
			if discarded.Valid() {
				t.Fatal("discarded array survived")
			}
			Scoped(func() { held.Detach(detached) })
			if detached.Valid() {
				t.Fatal("detached array survived the scope it was detached into")
			}
			if !panics(func() { other.Discard(kept) }) {
				t.Fatal("no panic discarding an array the scope does not hold")
			}
			if !panics(func() { other.Detach(kept) }) {
				t.Fatal("no panic detaching an array the scope does not hold")
			}
			if !panics(func() { other.Attach(kept) }) {
				t.Fatal("no panic holding an array another scope holds")
			}
			if !panics(func() { held.Attach(kept) }) {
				t.Fatal("no panic holding an array twice")
			}
		})
		if !kept.Valid() {
			t.Fatal("held array was freed with the scope that created it")
		}
		held.Close()
		if kept.Valid() {
			t.Fatal("held array survived its scope's close")
		}
	})
}

// A scope ends when its function panics, so whatever recovers is back in
// the scope it started from.
func TestScopeEndsOnPanic(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		start := currentScope
		var a *Array
		func() {
			defer func() { _ = recover() }()
			Scoped(func() {
				a = FromValue(1)
				panic("build failed")
			})
		}()
		if a.Valid() {
			t.Fatal("array survived the scope that panicked")
		}
		if currentScope != start {
			t.Fatal("registry not back in the caller's scope after a panic")
		}
	})
}

// Nothing built in a compile trace may outlive it: holding a trace array
// fails the compiled call.
func TestCompileTraceRefusesEscapes(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		held := NewScope()
		defer held.Close()
		double := Compile("scope_test_escape", func(in ...*Array) []*Array {
			out := in[0].Add(in[0])
			held.Attach(out)
			return []*Array{out}
		})
		defer func() {
			if recover() == nil {
				t.Fatal("no panic for an array held out of a compile trace")
			}
		}()
		Scoped(func() { double(FromValue(1)) })
	})
}

func panics(fn func()) (panicked bool) {
	defer func() { panicked = recover() != nil }()
	fn()
	return false
}

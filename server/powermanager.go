package server

import "sync"

// inferencePowerAssertion holds a system power assertion (prevent idle/system
// sleep; never display sleep) while local inference requests are in flight.
//
// It is refcounted so concurrent requests share a single OS assertion, and it
// is fail-open: if the OS refuses the assertion, requests proceed normally.
// Power management must never break inference (issue #4072).
type inferencePowerAssertion struct {
	mu        sync.Mutex
	holders   int
	doRelease func()
}

func (p *inferencePowerAssertion) acquire() func() {
	p.mu.Lock()
	if p.holders == 0 {
		if release, ok := platformAcquirePowerAssertion(); ok {
			p.doRelease = release
		}
	}
	p.holders++
	p.mu.Unlock()

	var released bool
	return func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		if released {
			return
		}
		released = true
		p.holders--
		if p.holders == 0 && p.doRelease != nil {
			release := p.doRelease
			p.doRelease = nil
			release()
		}
	}
}

// sharedInferencePower is process-wide: one assertion covers all concurrent
// local inference requests (generate/chat/embed/speech).
var sharedInferencePower inferencePowerAssertion

// holdPowerAssertionDuringInference acquires the shared assertion and returns
// a release func for defer. Call it after a runner has been scheduled
// successfully; the deferred release runs when the handler returns.
func holdPowerAssertionDuringInference() func() {
	return sharedInferencePower.acquire()
}

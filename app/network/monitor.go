package network

import (
	"context"
	"sync"
)

type ConnectivityStatus int

const (
	StatusUnknown ConnectivityStatus = iota
	StatusOnline
	StatusOffline
)

type ConnectivityChangeHandler func(status ConnectivityStatus)

type Monitor struct {
	mu       sync.RWMutex
	status   ConnectivityStatus
	handlers []ConnectivityChangeHandler
	stopChan chan struct{}
}

func NewMonitor() *Monitor {
	return &Monitor{
		status:   StatusUnknown,
		handlers: make([]ConnectivityChangeHandler, 0),
	}
}

func (m *Monitor) Start(ctx context.Context) {
	m.mu.Lock()
	if m.stopChan != nil {
		m.mu.Unlock()
		return
	}
	m.stopChan = make(chan struct{})
	m.mu.Unlock()

	m.startPlatformMonitor(ctx)
}

func (m *Monitor) checkConnectivity() {
	online := m.checkPlatformConnectivity()

	m.mu.Lock()
	oldStatus := m.status
	if online {
		m.status = StatusOnline
	} else {
		m.status = StatusOffline
	}
	handlers := m.handlers
	m.mu.Unlock()

	if oldStatus != m.status {
		for _, handler := range handlers {
			handler(m.status)
		}
	}
}

func (m *Monitor) OnConnectivityChange(handler ConnectivityChangeHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers = append(m.handlers, handler)
}

func (m *Monitor) IsOnline() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.status == StatusOnline
}

// Disconnected returns a channel that receives a signal when the network goes offline
func (m *Monitor) Disconnected() <-chan struct{} {
	ch := make(chan struct{})

	m.OnConnectivityChange(func(status ConnectivityStatus) {
		if status == StatusOffline {
			select {
			case ch <- struct{}{}:
			default:
				// Don't block if already signaled
			}
		}
	})

	return ch
}

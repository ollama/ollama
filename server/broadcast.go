package server

import (
	"sync"

	"github.com/ollama/ollama/api"
)

type Broadcaster struct {
	sync.RWMutex
	clients map[chan api.MonitorResponse]struct{}
}

func NewBroadcaster() *Broadcaster {
	return &Broadcaster{
		clients: make(map[chan api.MonitorResponse]struct{}),
	}
}

func (b *Broadcaster) Subscribe() chan api.MonitorResponse {
	b.Lock()
	defer b.Unlock()
	ch := make(chan api.MonitorResponse, 100)
	b.clients[ch] = struct{}{}
	return ch
}

func (b *Broadcaster) Unsubscribe(ch chan api.MonitorResponse) {
	b.Lock()
	defer b.Unlock()
	delete(b.clients, ch)
	close(ch)
}

func (b *Broadcaster) Broadcast(msg api.MonitorResponse) {
	b.RLock()
	defer b.RUnlock()
	for ch := range b.clients {
		select {
		case ch <- msg:
		default:
			// If a client is too slow and its channel buffer is full, drop the message
			// to avoid blocking the generator.
		}
	}
}

package ollamarunner

import (
	"container/heap"
	"sync"
	"time"
)

// RequestPriority defines the priority levels for requests
type RequestPriority int

const (
	PriorityLow RequestPriority = iota
	PriorityNormal
	PriorityHigh
)

// ScheduledRequest represents a request waiting to be scheduled
type ScheduledRequest struct {
	// Sequence is the request sequence
	sequence *Sequence

	// Priority of this request
	priority RequestPriority

	// Timestamp when request was added to queue
	addedAt time.Time

	// Estimated number of tokens to process
	estimatedTokens int

	// Current position in queue
	index int
}

// RequestQueue manages pending requests using a priority queue
type RequestQueue struct {
	mu     sync.Mutex
	queue  []*ScheduledRequest
	nextID int
}

// NewRequestQueue creates a new request queue
func NewRequestQueue() *RequestQueue {
	q := &RequestQueue{
		queue: make([]*ScheduledRequest, 0),
	}
	heap.Init(q)
	return q
}

// Len implements heap.Interface
func (q *RequestQueue) Len() int { return len(q.queue) }

// Less implements heap.Interface (higher priority first, then FIFO)
func (q *RequestQueue) Less(i, j int) bool {
	if q.queue[i].priority != q.queue[j].priority {
		return q.queue[i].priority > q.queue[j].priority
	}
	return q.queue[i].addedAt.Before(q.queue[j].addedAt)
}

// Swap implements heap.Interface
func (q *RequestQueue) Swap(i, j int) {
	q.queue[i], q.queue[j] = q.queue[j], q.queue[i]
	q.queue[i].index = i
	q.queue[j].index = j
}

// Push implements heap.Interface
func (q *RequestQueue) Push(x interface{}) {
	n := len(q.queue)
	item := x.(*ScheduledRequest)
	item.index = n
	q.queue = append(q.queue, item)
}

// Pop implements heap.Interface
func (q *RequestQueue) Pop() interface{} {
	old := q.queue
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // avoid memory leak
	item.index = -1 // for safety
	q.queue = old[0 : n-1]
	return item
}

// Enqueue adds a request to the queue
func (q *RequestQueue) Enqueue(seq *Sequence, priority RequestPriority, estimatedTokens int) {
	q.mu.Lock()
	defer q.mu.Unlock()

	req := &ScheduledRequest{
		sequence:        seq,
		priority:        priority,
		addedAt:         time.Now(),
		estimatedTokens: estimatedTokens,
	}
	heap.Push(q, req)
}

// Dequeue removes and returns the highest priority request
func (q *RequestQueue) Dequeue() *ScheduledRequest {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.Len() == 0 {
		return nil
	}
	return heap.Pop(q).(*ScheduledRequest)
}

// Peek returns the highest priority request without removing it
func (q *RequestQueue) Peek() *ScheduledRequest {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.Len() == 0 {
		return nil
	}
	return q.queue[0]
}

// Remove removes a specific request from the queue
func (q *RequestQueue) Remove(seq *Sequence) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for i, req := range q.queue {
		if req.sequence == seq {
			heap.Remove(q, i)
			return
		}
	}
}

// Size returns the current queue size
func (q *RequestQueue) Size() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.Len()
}

// SchedulerConfig holds configuration for the continuous batching scheduler
type SchedulerConfig struct {
	// MaxBatchSize is the maximum number of sequences in a batch
	MaxBatchSize int

	// MaxTokensPerBatch is the maximum total tokens (prompt + generated) per batch
	MaxTokensPerBatch int

	// ScheduleInterval is how often to check for new requests to add
	ScheduleInterval time.Duration

	// PreemptThreshold allows preemption if a batch has less than this % of max capacity
	PreemptThreshold float64
}

// ContinuousBatchScheduler manages dynamic batching of requests
type ContinuousBatchScheduler struct {
	config     SchedulerConfig
	queue      *RequestQueue
	activeSeqs map[*Sequence]struct{}

	// For tracking batch state
	batchTokens int // Total tokens in current batch

	mu sync.Mutex
}

// NewContinuousBatchScheduler creates a new continuous batching scheduler
func NewContinuousBatchScheduler(config SchedulerConfig) *ContinuousBatchScheduler {
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 32
	}
	if config.MaxTokensPerBatch <= 0 {
		config.MaxTokensPerBatch = 8192
	}
	if config.ScheduleInterval <= 0 {
		config.ScheduleInterval = 10 * time.Millisecond
	}
	if config.PreemptThreshold <= 0 {
		config.PreemptThreshold = 0.8
	}

	return &ContinuousBatchScheduler{
		config:     config,
		queue:      NewRequestQueue(),
		activeSeqs: make(map[*Sequence]struct{}),
	}
}

// AddRequest adds a new request to the scheduling queue
func (s *ContinuousBatchScheduler) AddRequest(seq *Sequence, priority RequestPriority, estimatedTokens int) {
	s.queue.Enqueue(seq, priority, estimatedTokens)
}

// RemoveRequest removes a request from the queue (e.g., if cancelled)
func (s *ContinuousBatchScheduler) RemoveRequest(seq *Sequence) {
	s.queue.Remove(seq)
	s.mu.Lock()
	delete(s.activeSeqs, seq)
	s.mu.Unlock()
}

// GetNextRequests returns requests that should be added to the current batch
// Returns up to n requests
func (s *ContinuousBatchScheduler) GetNextRequests(maxSeqs int, maxTokens int) []*Sequence {
	s.mu.Lock()
	defer s.mu.Unlock()

	var seqs []*Sequence

	// Check how many slots we have available
	availableSlots := maxSeqs - len(s.activeSeqs)
	if availableSlots <= 0 {
		return seqs
	}

	// Get pending requests from queue
	for i := 0; i < availableSlots; i++ {
		req := s.queue.Dequeue()
		if req == nil {
			break
		}

		seqs = append(seqs, req.sequence)
		s.activeSeqs[req.sequence] = struct{}{}
	}

	return seqs
}

// MarkSequenceComplete marks a sequence as complete and removes it from active set
func (s *ContinuousBatchScheduler) MarkSequenceComplete(seq *Sequence) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.activeSeqs, seq)
}

// ActiveSequenceCount returns the number of currently active sequences
func (s *ContinuousBatchScheduler) ActiveSequenceCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.activeSeqs)
}

// QueuedRequestCount returns the number of queued (waiting) requests
func (s *ContinuousBatchScheduler) QueuedRequestCount() int {
	return s.queue.Size()
}

// ShouldScheduleNewBatch returns true if we should schedule a new batch
// This happens when:
// 1. There are queued requests AND
// 2. Either (no active sequences) OR (batch has room for more)
func (s *ContinuousBatchScheduler) ShouldScheduleNewBatch() bool {
	activeCount := s.ActiveSequenceCount()
	queueCount := s.QueuedRequestCount()

	// Schedule if we have queued requests and either no active sequences or room for more
	return queueCount > 0 && (activeCount == 0 || activeCount < s.config.MaxBatchSize)
}

// CanAddToBatch returns true if we can add more sequences to the current batch
func (s *ContinuousBatchScheduler) CanAddToBatch(currentBatchSize int) bool {
	return currentBatchSize < s.config.MaxBatchSize && s.queue.Size() > 0
}

// UpdateBatchTokens updates the tracked token count for the current batch
func (s *ContinuousBatchScheduler) UpdateBatchTokens(tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.batchTokens = tokens
}

// GetBatchUtilization returns the utilization of the current batch (0-1)
func (s *ContinuousBatchScheduler) GetBatchUtilization() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.config.MaxBatchSize == 0 {
		return 0
	}

	activeCount := float64(len(s.activeSeqs))
	return activeCount / float64(s.config.MaxBatchSize)
}

package ml

// Scheduler is an interface that can be implemented by a Backend to schedule resources.
type Scheduler interface {
	Schedule()
}

// Reserver is an optional interface that can be implemented by a Scheduler to reserve resources for the compute graph.
type Reserver interface {
	Reserve()
}

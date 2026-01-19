package imagegen

// ProgressFunc is called during generation with step progress.
type ProgressFunc func(step, totalSteps int)

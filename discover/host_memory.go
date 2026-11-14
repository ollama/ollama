package discover

// MemoryHeadroom separates reclaimable capacity from immediate physical scarcity.
// Compressed is the original byte size of content held in the host compressor.
// It is a host-wide pressure observation, not this process's allocation deficit.
type MemoryHeadroom struct {
	Available  int64
	Immediate  int64 // Negative when below the OS free-page target.
	Compressed int64
}

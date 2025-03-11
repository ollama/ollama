package input

// Input represents one token in the input stream
type Input struct {
	// Token is a single element of text.
	Token int32

	// Multimodal is opaque data representing a non-text
	// element such as an image (or part of one if the image
	// can be processed in pieces). It may be either together
	// with Token or on its own.
	Multimodal any

	// MultimodalHash is a unique representation of the data
	// stored in Multimodal, used for caching and comparing
	// equality.
	MultimodalHash uint64

	// BatchBreak forces a new batch to be started with this
	// input. For example, this can be used to align images
	// with batches. Note that batches may be divided in additional
	// locations as well.
	BatchBreak bool
}

// MultimodalIndex is a multimodal element (such as an image)
// together with an index into the slice of Inputs with the
// corresponding token. Note that the index is not the same
// as the position - to find that use the index with the
// Positions slice.
type MultimodalIndex struct {
	Index      int
	Multimodal any
}

// Options contains the inputs for a model forward pass
type Options struct {
	Inputs     []int32
	Multimodal []MultimodalIndex
	Positions  []int32
	Sequences  []int
	Outputs    []int32
}

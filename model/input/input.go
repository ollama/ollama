package input

import "github.com/ollama/ollama/ml"

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

	// SameBatch forces the following number of tokens to be processed
	// in a single batch, breaking and extending batches as needed.
	// Useful for things like images that must be processed in one
	// shot.
	SameBatch int
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

// Batch contains the inputs for a model forward pass
type Batch struct {
	// Inputs is the input tokens, including placeholders for multimodal inputs.
	Inputs ml.Tensor

	// Multimodal is a set of multimodal embeddings previously created by
	// EncodeMultimodal, along with an index into Inputs. Unused for text-only
	// models or for batches without multimodal elements.
	Multimodal []MultimodalIndex

	// Positions is the position for each Input, relative to its sequence. Equal
	// in length to Inputs.
	Positions []int32

	// Sequences is the sequence for each Input. Equal in length to Inputs.
	Sequences []int

	// Outputs are the set of indicies into Inputs for which output data should
	// be returned.
	Outputs []int32
}

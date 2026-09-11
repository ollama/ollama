package input

import "github.com/ollama/ollama/x/ml"

// Multimodal is a multimodal embedding or a component of one.
// For example, it could be a row of an image that can be processed
// independently.
type Multimodal struct {
	// Tensor is the embedding data. Implementations may chose what to
	// store here or it may be nil if not needed. However, any ml.Tensor
	// objects must be stored here and not in Data.
	Tensor ml.Tensor

	// Data is implementation-specific opaque data, such as metadata on how
	// to layout Tensor. It may be nil if not needed. It may also store larger
	// objects such as complete images if they are to be processed later.
	Data any
}

// Input represents one token in the input stream
type Input struct {
	// Token is a single element of text.
	Token int32

	// Multimodal is represents a non-text element such as an
	// image (or part of one if the image can be processed in pieces).
	// It may be used either together with Token or on its own.
	Multimodal []Multimodal

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
	Multimodal []Multimodal
}

// Batch contains the inputs for a model forward pass
type Batch struct {
	// Inputs is the input tokens, including placeholders for multimodal inputs.
	Inputs ml.Tensor

	// Outputs are the set of indicies into Inputs for which output data should
	// be returned.
	Outputs ml.Tensor

	// TODO maybe not the optimal way to handle this
	// Offset of final tensor in the final batch
	Offset int

	// Positions is the position for each Input, relative to its sequence. Equal
	// in length to Inputs.
	Positions []int32

	// Sequences is the sequence for each Input. Equal in length to Inputs.
	Sequences []int

	// Multimodal is a set of multimodal embeddings previously created by
	// EncodeMultimodal, along with an index into Inputs. Unused for text-only
	// models or for batches without multimodal elements.
	Multimodal []MultimodalIndex
}

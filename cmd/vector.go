package cmd

import (
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Vector struct {
	Data          *mat.VecDense // the embedding vector
	UserInput     string        // the user input segment of the text
	ModelResponse string        // the model response segment of the text
}

// VectorSimilarity is a vector and its similarity to another vector
type VectorSimilarity struct {
	Vector     Vector
	Similarity float64
}

type BySimilarity []VectorSimilarity

func (a BySimilarity) Len() int           { return len(a) }
func (a BySimilarity) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a BySimilarity) Less(i, j int) bool { return a[i].Similarity > a[j].Similarity }

// cosineSimilarity is a measure that calculates the cosine of the angle between two vectors.
// This value will range from -1 to 1, where 1 means the vectors are identical.
func cosineSimilarity(vec1, vec2 *mat.VecDense) float64 {
	dotProduct := mat.Dot(vec1, vec2)
	norms := mat.Norm(vec1, 2) * mat.Norm(vec2, 2)

	if norms == 0 {
		return 0
	}
	return dotProduct / norms
}

type VectorSlice []Vector

func (vs *VectorSlice) Add(v Vector) {
	*vs = append(*vs, v)
}

func (vs *VectorSlice) Length() int {
	return len(*vs)
}

func (vs *VectorSlice) NearestNeighbors(embedding *mat.VecDense, n int) VectorSlice {
	if vs.Length() == 0 {
		return VectorSlice{}
	}
	similarities := make([]VectorSimilarity, vs.Length())
	for i, v := range *vs {
		similarity := cosineSimilarity(embedding, v.Data)
		similarities[i] = VectorSimilarity{Vector: v, Similarity: similarity}
	}
	sort.Sort(BySimilarity(similarities))
	if len(similarities) < n {
		n = len(similarities)
	}
	result := make(VectorSlice, n)
	for i := 0; i < n; i++ {
		result[i] = similarities[i].Vector
	}
	return result
}

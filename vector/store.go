package vector

import (
	"container/heap"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Embedding struct {
	Vector []float64 // the embedding vector
	Data   string    // the data represted by the embedding
}

type EmbeddingSimilarity struct {
	Embedding  Embedding // the embedding that was used to calculate the similarity
	Similarity float64   // the similarity between the embedding and the query
}

type Heap []EmbeddingSimilarity

func (h Heap) Len() int           { return len(h) }
func (h Heap) Less(i, j int) bool { return h[i].Similarity < h[j].Similarity }
func (h Heap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *Heap) Push(e any) {
	*h = append(*h, e.(EmbeddingSimilarity))
}

func (h *Heap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

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

func TopK(k int, query *mat.VecDense, embeddings []Embedding) []EmbeddingSimilarity {
	h := &Heap{}
	heap.Init(h)
	for _, emb := range embeddings {
		similarity := cosineSimilarity(query, mat.NewVecDense(len(emb.Vector), emb.Vector))
		heap.Push(h, EmbeddingSimilarity{Embedding: emb, Similarity: similarity})
		if h.Len() > k {
			heap.Pop(h)
		}
	}

	topK := make([]EmbeddingSimilarity, 0, h.Len())
	for h.Len() > 0 {
		topK = append(topK, heap.Pop(h).(EmbeddingSimilarity))
	}
	sort.Slice(topK, func(i, j int) bool {
		return topK[i].Similarity > topK[j].Similarity
	})

	return topK
}

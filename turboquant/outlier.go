package turboquant

import (
	"cmp"
	"slices"
)

// OutlierSplit partitions a vector's channels into outlier and regular sets.
// OutlierIndices and RegularIndices are sorted in ascending order.
// OutlierValues and RegularValues contain the corresponding channel values.
type OutlierSplit struct {
	OutlierIndices []uint16
	OutlierValues  []float32
	RegularIndices []uint16
	RegularValues  []float32
}

// SplitOutlierChannels identifies the top-outlierCount channels by absolute
// magnitude and returns them separately from the remaining regular channels.
// Tie-breaking is by index ascending for determinism.
// If outlierCount <= 0, all channels are returned as regular (empty outlier).
// If outlierCount >= len(values), all channels are returned as outlier.
func SplitOutlierChannels(values []float32, outlierCount int) OutlierSplit {
	dim := len(values)
	if outlierCount <= 0 || dim == 0 {
		indices := make([]uint16, dim)
		vals := make([]float32, dim)
		for i := range values {
			indices[i] = uint16(i)
			vals[i] = values[i]
		}
		return OutlierSplit{RegularIndices: indices, RegularValues: vals}
	}
	if outlierCount >= dim {
		indices := make([]uint16, dim)
		vals := make([]float32, dim)
		for i := range values {
			indices[i] = uint16(i)
			vals[i] = values[i]
		}
		return OutlierSplit{OutlierIndices: indices, OutlierValues: vals}
	}

	// Sort channel indices by abs value descending, tie-break by index ascending.
	order := make([]int, dim)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		va := abs32(values[a])
		vb := abs32(values[b])
		if va != vb {
			return cmp.Compare(vb, va)
		}
		return cmp.Compare(a, b)
	})

	outlierSet := make([]int, outlierCount)
	regularSet := make([]int, dim-outlierCount)
	copy(outlierSet, order[:outlierCount])
	copy(regularSet, order[outlierCount:])

	// Each set is sorted by index ascending so ChannelIndices is deterministic.
	slices.Sort(outlierSet)
	slices.Sort(regularSet)

	outlierIndices := make([]uint16, outlierCount)
	outlierValues := make([]float32, outlierCount)
	for i, idx := range outlierSet {
		outlierIndices[i] = uint16(idx)
		outlierValues[i] = values[idx]
	}

	regularCount := dim - outlierCount
	regularIndices := make([]uint16, regularCount)
	regularValues := make([]float32, regularCount)
	for i, idx := range regularSet {
		regularIndices[i] = uint16(idx)
		regularValues[i] = values[idx]
	}

	return OutlierSplit{
		OutlierIndices: outlierIndices,
		OutlierValues:  outlierValues,
		RegularIndices: regularIndices,
		RegularValues:  regularValues,
	}
}

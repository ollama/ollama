package main

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

type TensorRename struct {
	Role       string
	Left       string
	Right      string
	Confidence string // payload or descriptor
	SHA256     string
}

type ExpertFusion struct {
	Role       string
	Left       []string
	Right      []string
	LeftDType  string
	RightDType string
	LeftShape  []uint64
	RightShape []uint64
}

func detectExpertFusions(changes []TensorChange) []ExpertFusion {
	byKey := make(map[string]TensorChange, len(changes))
	for _, change := range changes {
		byKey[tensorKey(change.Role, change.Name)] = change
	}

	var fusions []ExpertFusion
	for _, direction := range []struct {
		sourceStatus, targetStatus string
		source, target             func(TensorChange) *Tensor
		reverse                    bool
	}{
		{"removed", "added", func(d TensorChange) *Tensor { return d.Left }, func(d TensorChange) *Tensor { return d.Right }, false},
		{"added", "removed", func(d TensorChange) *Tensor { return d.Right }, func(d TensorChange) *Tensor { return d.Left }, true},
	} {
		groups := make(map[string][]expertTensor)
		for _, change := range changes {
			if change.Status != direction.sourceStatus {
				continue
			}
			group, index, projection, ok := splitExpertName(change.Name)
			if !ok {
				continue
			}
			key := tensorKey(change.Role, group+"."+projection+".weight")
			groups[key] = append(groups[key], expertTensor{index: index, change: change})
		}
		for _, targetKey := range unionKeys(groups, nil) {
			targetChange, ok := byKey[targetKey]
			if !ok || targetChange.Status != direction.targetStatus {
				continue
			}
			experts := groups[targetKey]
			slices.SortFunc(experts, func(a, b expertTensor) int { return a.index - b.index })
			if !validExpertStack(experts, direction.source, direction.target(targetChange)) {
				continue
			}
			sources := make([]string, len(experts))
			for i, expert := range experts {
				sources[i] = expert.change.Name
			}
			target := direction.target(targetChange)
			base := direction.source(experts[0].change)
			fusion := ExpertFusion{
				Role:       target.Role,
				Left:       sources,
				Right:      []string{target.Name},
				LeftDType:  base.ModelDType,
				RightDType: target.ModelDType,
				LeftShape:  append([]uint64{uint64(len(experts))}, logicalShape(base)...),
				RightShape: logicalShape(target),
			}
			if direction.reverse {
				fusion.Left, fusion.Right = fusion.Right, fusion.Left
				fusion.LeftDType, fusion.RightDType = fusion.RightDType, fusion.LeftDType
				fusion.LeftShape, fusion.RightShape = fusion.RightShape, fusion.LeftShape
			}
			fusions = append(fusions, fusion)
		}
	}
	slices.SortFunc(fusions, func(a, b ExpertFusion) int {
		return strings.Compare(strings.Join(a.Left, "\x00"), strings.Join(b.Left, "\x00"))
	})
	return fusions
}

type expertTensor struct {
	index  int
	change TensorChange
}

func splitExpertName(name string) (group string, index int, projection string, ok bool) {
	if !strings.HasSuffix(name, ".weight") {
		return "", 0, "", false
	}
	for _, marker := range []string{".mlp.experts.", ".mlp.shared_experts.", ".mlp.switch_mlp.", ".moe.experts.", ".mixer.experts.", ".mixer.shared_experts."} {
		at := strings.Index(name, marker)
		if at < 0 {
			continue
		}
		rest := strings.TrimSuffix(name[at+len(marker):], ".weight")
		indexText, projection, found := strings.Cut(rest, ".")
		if !found || projection == "" {
			return "", 0, "", false
		}
		index, err := strconv.Atoi(indexText)
		if err != nil || index < 0 {
			return "", 0, "", false
		}
		return name[:at] + strings.TrimSuffix(marker, "."), index, projection, true
	}
	return "", 0, "", false
}

func validExpertStack(experts []expertTensor, source func(TensorChange) *Tensor, target *Tensor) bool {
	if len(experts) < 2 || target == nil {
		return false
	}
	base := source(experts[0].change)
	if base == nil {
		return false
	}
	for i, expert := range experts {
		tensor := source(expert.change)
		if expert.index != i || tensor == nil || tensor.ModelDType != base.ModelDType || !slices.Equal(logicalShape(tensor), logicalShape(base)) {
			return false
		}
	}
	want := append([]uint64{uint64(len(experts))}, logicalShape(base)...)
	return slices.Equal(logicalShape(target), want)
}

func logicalShape(t *Tensor) []uint64 {
	if t.Quantization != nil {
		return t.Quantization.LogicalShape
	}
	return t.Shape
}

func detectRenames(changes []TensorChange, fusions []ExpertFusion) []TensorRename {
	excluded := make(map[string]bool)
	for _, fusion := range fusions {
		for _, name := range append(slices.Clone(fusion.Left), fusion.Right...) {
			excluded[tensorKey(fusion.Role, name)] = true
		}
	}
	left, right := make(map[string][]TensorChange), make(map[string][]TensorChange)
	for _, change := range changes {
		if excluded[tensorKey(change.Role, change.Name)] {
			continue
		}
		switch change.Status {
		case "removed":
			if change.Left.CompanionOf == "" {
				left[renameSignature(change.Left)] = append(left[renameSignature(change.Left)], change)
			}
		case "added":
			if change.Right.CompanionOf == "" {
				right[renameSignature(change.Right)] = append(right[renameSignature(change.Right)], change)
			}
		}
	}

	var renames []TensorRename
	for _, signature := range unionKeys(left, right) {
		removed, added := left[signature], right[signature]
		if len(removed) == 0 || len(added) == 0 {
			continue
		}
		usedLeft, usedRight := make(map[int]bool), make(map[int]bool)
		for i, l := range removed {
			if l.Left.SHA256 == "" {
				continue
			}
			var candidates []int
			for j, r := range added {
				if !usedRight[j] && l.Left.SHA256 == r.Right.SHA256 {
					candidates = append(candidates, j)
				}
			}
			if j, ok := uniqueBestName(l.Name, added, candidates); ok {
				usedLeft[i], usedRight[j] = true, true
				renames = append(renames, TensorRename{Role: l.Role, Left: l.Name, Right: added[j].Name, Confidence: "payload", SHA256: l.Left.SHA256})
			}
		}
		for i, l := range removed {
			if usedLeft[i] {
				continue
			}
			var candidates []int
			for j := range added {
				if !usedRight[j] {
					candidates = append(candidates, j)
				}
			}
			if j, ok := uniqueBestName(l.Name, added, candidates); ok && (len(candidates) == 1 || nameSimilarity(l.Name, added[j].Name) >= 20) {
				usedRight[j] = true
				renames = append(renames, TensorRename{Role: l.Role, Left: l.Name, Right: added[j].Name, Confidence: "descriptor"})
			}
		}
	}
	slices.SortFunc(renames, func(a, b TensorRename) int { return strings.Compare(a.Role+"/"+a.Left, b.Role+"/"+b.Left) })
	return renames
}

func renameSignature(t *Tensor) string {
	return fmt.Sprintf("%s\x00%s\x00%s\x00%v\x00%d", t.Role, t.ModelDType, t.DType, logicalShape(t), t.Bytes)
}

func uniqueBestName(name string, candidates []TensorChange, indexes []int) (int, bool) {
	if len(indexes) == 1 {
		return indexes[0], true
	}
	best, bestScore, tied := -1, -1, false
	for _, index := range indexes {
		score := nameSimilarity(name, candidates[index].Name)
		switch {
		case score > bestScore:
			best, bestScore, tied = index, score, false
		case score == bestScore:
			tied = true
		}
	}
	return best, best >= 0 && !tied
}

func nameSimilarity(a, b string) int {
	aa, bb := strings.Split(a, "."), strings.Split(b, ".")
	prefix := 0
	for prefix < min(len(aa), len(bb)) && aa[prefix] == bb[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < min(len(aa), len(bb)) && aa[len(aa)-1-suffix] == bb[len(bb)-1-suffix] {
		suffix++
	}
	return prefix + 10*suffix
}

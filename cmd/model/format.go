package main

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	ollamaformat "github.com/ollama/ollama/format"
	"github.com/pmezard/go-difflib/difflib"
)

// WriteText emits deterministic diff-style text. Limits affect display only;
// Compare always retains all selected entries and metadata changes.
func WriteText(w io.Writer, r *Report, all, summaryOnly bool, limit int) error {
	if err := writeTextHeader(w, r.Left, r.Right); err != nil {
		return err
	}
	return writeTextBody(w, r, all, summaryOnly, limit)
}

func writeTextHeader(w io.Writer, left, right Source) error {
	_, err := fmt.Fprintf(w, "--- %s\n+++ %s\n", sourceLabel(left), sourceLabel(right))
	return err
}

func writeTextBody(w io.Writer, r *Report, all, summaryOnly bool, limit int) error {
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n\n", headline(r))
	writeSummary(&b, r)
	writeStats(&b, r, all, summaryOnly, limit)
	if !summaryOnly {
		writeAssociations(&b, r, all, limit)
	}

	if !summaryOnly {
		shown := 0
		associated := associationKeys(r)
		if len(r.Metadata) > 0 {
			fmt.Fprintln(&b, "\nMetadata\n========")
		}
		for _, d := range r.Metadata {
			if !all && shown >= limit {
				continue
			}
			writeMetadata(&b, d, all)
			shown++
		}

		changedTensors := 0
		for _, d := range r.Tensors {
			if all || len(d.Changes) > 0 && !associated[tensorKey(d.Role, d.Name)] {
				changedTensors++
			}
		}
		if changedTensors > 0 && (all || shown < limit) {
			fmt.Fprintln(&b, "\nTensors\n=======")
		}
		for _, d := range r.Tensors {
			if !all && len(d.Changes) == 0 {
				continue
			}
			if !all && associated[tensorKey(d.Role, d.Name)] {
				continue
			}
			if !all && shown >= limit {
				continue
			}
			writeTensor(&b, d, all)
			shown++
		}

		omitted := len(r.Metadata) + changedTensors - shown
		if omitted > 0 {
			fmt.Fprintf(&b, "\n... %d %s omitted; use --all.\n", omitted, plural(omitted, "change", "changes"))
		}
	}

	for _, warning := range r.Warnings {
		fmt.Fprintf(&b, "\nNote: %s\n", warning)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func associationKeys(r *Report) map[string]bool {
	keys := make(map[string]bool)
	for _, rename := range r.Renames {
		keys[tensorKey(rename.Role, rename.Left)] = true
		keys[tensorKey(rename.Role, rename.Right)] = true
	}
	for _, fusion := range r.ExpertFusions {
		for _, name := range fusion.Left {
			keys[tensorKey(fusion.Role, name)] = true
		}
		for _, name := range fusion.Right {
			keys[tensorKey(fusion.Role, name)] = true
		}
	}
	return keys
}

func sourceLabel(s Source) string {
	parts := []string{s.Format}
	if s.Digest != "" {
		parts = append(parts, "manifest "+shortHash(s.Digest))
	}
	return fmt.Sprintf("%s [%s]", terminalText(s.Reference), strings.Join(parts, "; "))
}

func writeSummary(b *strings.Builder, r *Report) {
	fmt.Fprintln(b, "Summary\n=======")
	fmt.Fprintf(b, "Scope: %s\n", scopeText(r.Scope))
	if r.Filter != "" {
		fmt.Fprintf(b, "Tensor filter: %s (known companions included)\n", terminalText(r.Filter))
	}

	s := r.Summary
	fmt.Fprintf(b, "Metadata: %s (semantic %d, provenance %d)\n", changeSummary(s.MetadataChanges), s.SemanticMetadata, s.ProvenanceMetadata)
	fmt.Fprintf(b, "Tensors: %d total; %s\n", s.Total, countSummary(s.Counts))
	fmt.Fprintf(b, "Tensor data: %s -> %s\n", ollamaformat.HumanBytes(s.LeftBytes), ollamaformat.HumanBytes(s.RightBytes))
	if s.LeftBlobs > 0 || s.RightBlobs > 0 {
		fmt.Fprintf(b, "Tensor blobs: %d -> %d", s.LeftBlobs, s.RightBlobs)
		if s.SharedBlobs > 0 {
			fmt.Fprintf(b, "; %d shared by digest", s.SharedBlobs)
		}
		fmt.Fprintln(b)
	}
	if s.LeftFiles > 0 || s.RightFiles > 0 {
		fmt.Fprintf(b, "Tensor files outside the blob store: %d -> %d\n", s.LeftFiles, s.RightFiles)
	}

	var identity []string
	if s.BlobMatched > 0 {
		identity = append(identity, fmt.Sprintf("%d %s matched by blob digest", s.BlobMatched, plural(s.BlobMatched, "tensor", "tensors")))
	}
	if s.SameFileMatched > 0 {
		identity = append(identity, fmt.Sprintf("%d %s matched by the same file range", s.SameFileMatched, plural(s.SameFileMatched, "tensor", "tensors")))
	}
	if s.PayloadHashed > 0 {
		identity = append(identity, fmt.Sprintf("%d %s compared by SHA-256 (%s read)", s.PayloadHashed, plural(s.PayloadHashed, "tensor", "tensors"), ollamaformat.HumanBytes(s.BytesHashed)))
	}
	if len(identity) > 0 {
		fmt.Fprintln(b, "Payload identity:")
		for _, line := range identity {
			fmt.Fprintf(b, "  %s\n", line)
		}
	}
	if s.LayoutChanges > 0 {
		fmt.Fprintf(b, "Storage references changed for %d %s\n", s.LayoutChanges, plural(s.LayoutChanges, "tensor", "tensors"))
	}
	if s.PayloadUnchecked > 0 {
		fmt.Fprintf(b, "WARNING: payload data was not checked for %d %s.\n", s.PayloadUnchecked, plural(s.PayloadUnchecked, "tensor", "tensors"))
	}

	writeGroups(b, "Components", s.Components)
	writeGroups(b, "Model dtypes", s.ModelDTypes)
	writeGroups(b, "Storage encodings", s.StorageDTypes)
}

func headline(r *Report) string {
	s := r.Summary
	changes := s.Changed + s.Added + s.Removed
	line := fmt.Sprintf("%d %s", changes, plural(changes, "change", "changes"))
	payloadOnly := 0
	for _, count := range s.PayloadOnlyDTypes {
		payloadOnly += count
	}
	if changes > 0 && payloadOnly == changes {
		line += ": all payload-only"
		dtypes := unionKeys(s.PayloadOnlyDTypes, nil)
		if len(dtypes) > 0 {
			line += " within " + strings.Join(dtypes, ", ")
		}
	} else if payloadOnly > 0 {
		line += fmt.Sprintf(": %d payload-only", payloadOnly)
	}
	line = fmt.Sprintf("%s; %d descriptor %s; %d dtype %s; metadata semantic %d / provenance %d",
		line,
		s.DescriptorChanges, plural(s.DescriptorChanges, "change", "changes"),
		s.DTypeTransitions, plural(s.DTypeTransitions, "transition", "transitions"),
		s.SemanticMetadata, s.ProvenanceMetadata)
	if len(r.Renames) > 0 {
		line += fmt.Sprintf("; %d rename %s", len(r.Renames), plural(len(r.Renames), "candidate", "candidates"))
	}
	if len(r.ExpertFusions) > 0 {
		line += fmt.Sprintf("; %d expert-fusion %s", len(r.ExpertFusions), plural(len(r.ExpertFusions), "mapping", "mappings"))
	}
	return line
}

func writeAssociations(b *strings.Builder, r *Report, all bool, limit int) {
	if len(r.Renames) > 0 {
		fmt.Fprintln(b, "\nRename candidates\n=================")
		for i, rename := range r.Renames {
			if !all && i >= limit {
				fmt.Fprintf(b, "\n... %d rename candidates omitted; use --all.\n", len(r.Renames)-i)
				break
			}
			fmt.Fprintf(b, "\n--- tensor/%s/%s\n+++ tensor/%s/%s\n", rename.Role, terminalText(rename.Left), rename.Role, terminalText(rename.Right))
			if rename.Confidence == "payload" {
				fmt.Fprintf(b, "  same descriptor and payload sha256: %s\n", rename.SHA256)
			} else {
				fmt.Fprintln(b, "! same model dtype, storage encoding, logical shape, and byte size; payload differs")
			}
		}
	}
	if len(r.ExpertFusions) > 0 {
		fmt.Fprintln(b, "\nExpert fusion\n=============")
		for i, fusion := range r.ExpertFusions {
			if !all && i >= limit {
				fmt.Fprintf(b, "\n... %d expert-fusion mappings omitted; use --all.\n", len(r.ExpertFusions)-i)
				break
			}
			fmt.Fprintf(b, "\n--- %s\n+++ %s\n", tensorSetLabel(fusion.Role, fusion.Left), tensorSetLabel(fusion.Role, fusion.Right))
			if fusion.LeftDType == fusion.RightDType {
				fmt.Fprintf(b, "  model dtype: %s\n", fusion.LeftDType)
			} else {
				fmt.Fprintf(b, "- model dtype: %s\n+ model dtype: %s\n", fusion.LeftDType, fusion.RightDType)
			}
			fmt.Fprintf(b, "  logical shape: %s\n", shapeText(fusion.LeftShape))
			fmt.Fprintf(b, "- topology: %s\n", expertTopology(fusion.Left))
			fmt.Fprintf(b, "+ topology: %s\n", expertTopology(fusion.Right))
			fmt.Fprintln(b, "  mapping: recognized from complete expert indices and logical shape")
		}
	}
}

func expertTopology(names []string) string {
	if len(names) == 1 {
		return "1 stacked tensor"
	}
	return fmt.Sprintf("%d per-expert tensors", len(names))
}

func writeStats(b *strings.Builder, r *Report, all, summaryOnly bool, limit int) {
	if r.Stats == nil {
		return
	}
	s := r.Stats
	fmt.Fprintln(b, "\nQuantization statistics\n=======================")
	fmt.Fprintf(b, "Data inspected: %s", ollamaformat.HumanBytes(s.BytesRead))
	if s.ExtraBytesRead > 0 {
		fmt.Fprintf(b, " (%s beyond required payload hashing)", ollamaformat.HumanBytes(s.ExtraBytesRead))
	}
	fmt.Fprintln(b)
	writeQuantStatsLine(b, "left", s.Left)
	writeQuantStatsLine(b, "right", s.Right)

	if len(s.Comparisons) == 0 {
		fmt.Fprintln(b, "Dequantized NMSE: no changed, shape-aligned tensor pairs used supported numeric encodings.")
		return
	}
	var squaredError, leftEnergy, tensorNMSE float64
	var values, nonFinite uint64
	for _, comparison := range s.Comparisons {
		squaredError += comparison.SquaredErr
		leftEnergy += comparison.LeftEnergy
		values += comparison.Values
		nonFinite += comparison.NonFinite
		tensorNMSE += comparison.NMSE
	}
	nmse := squaredError / leftEnergy
	if leftEnergy == 0 {
		if squaredError == 0 {
			nmse = 0
		} else {
			nmse = math.Inf(1)
		}
	}
	fmt.Fprintf(b, "Dequantized NMSE (right vs left): aggregate %.4e; mean tensor %.4e across %d tensors and %d finite values", nmse, tensorNMSE/float64(len(s.Comparisons)), len(s.Comparisons), values)
	if nonFinite > 0 {
		fmt.Fprintf(b, "; %d non-finite pairs excluded", nonFinite)
	}
	fmt.Fprintln(b)
	if summaryOnly {
		return
	}

	comparisons := slices.Clone(s.Comparisons)
	slices.SortFunc(comparisons, func(a, b NumericComparison) int { return cmp.Compare(b.NMSE, a.NMSE) })
	show := min(len(comparisons), limit, 10)
	if all {
		show = len(comparisons)
	}
	for _, comparison := range comparisons[:show] {
		fmt.Fprintf(b, "  %s/%s: %.4e (%s -> %s; %d values)\n", comparison.Role, terminalText(comparison.Name), comparison.NMSE, comparison.LeftDType, comparison.RightDType, comparison.Values)
	}
	if show < len(comparisons) {
		fmt.Fprintf(b, "  ... %d tensor statistics omitted; use --all.\n", len(comparisons)-show)
	}
}

func writeQuantStatsLine(b *strings.Builder, side string, stats QuantStats) {
	var types []string
	for _, entry := range []struct {
		name  string
		count int
	}{{"MXFP8", stats.MXFP8Tensors}, {"MXFP4", stats.MXFP4Tensors}, {"NVFP4", stats.NVFP4Tensors}} {
		if entry.count > 0 {
			types = append(types, fmt.Sprintf("%s %d", entry.name, entry.count))
		}
	}
	if len(types) == 0 {
		types = append(types, "no MXFP/NVFP tensors")
	}
	fmt.Fprintf(b, "%s: %s", side, strings.Join(types, ", "))
	if stats.E4M3PayloadBlocks > 0 {
		rate := percent(stats.E4M3SaturatedBlocks, stats.E4M3PayloadBlocks)
		fmt.Fprintf(b, "; E4M3-max blocks %.3f%%", rate)
		if rate > 20 {
			fmt.Fprint(b, " [clipping signature]")
		}
	}
	if stats.E8M0Scales > 0 {
		fmt.Fprintf(b, "; E8M0 max %.3f%%", percent(stats.E8M0MaxScales, stats.E8M0Scales))
		if stats.E8M0InvalidScales > 0 {
			fmt.Fprintf(b, ", invalid %d", stats.E8M0InvalidScales)
		}
	}
	if stats.E4M3Scales > 0 {
		fmt.Fprintf(b, "; E4M3 scale max %.3f%%", percent(stats.E4M3MaxScales, stats.E4M3Scales))
		if stats.E4M3InvalidScales > 0 {
			fmt.Fprintf(b, ", invalid %d", stats.E4M3InvalidScales)
		}
	}
	fmt.Fprintln(b)
}

func percent(part, total uint64) float64 {
	if total == 0 {
		return 0
	}
	return 100 * float64(part) / float64(total)
}

func tensorSetLabel(role string, names []string) string {
	if len(names) == 1 {
		return "tensor/" + role + "/" + terminalText(names[0])
	}
	firstGroup, firstIndex, firstProjection, firstOK := splitExpertName(names[0])
	lastGroup, lastIndex, lastProjection, lastOK := splitExpertName(names[len(names)-1])
	if firstOK && lastOK && firstGroup == lastGroup && firstProjection == lastProjection {
		return fmt.Sprintf("tensor/%s/%s.{%d..%d}.%s.weight", role, terminalText(firstGroup), firstIndex, lastIndex, terminalText(firstProjection))
	}
	return fmt.Sprintf("tensor-set/%s/%d-tensors", role, len(names))
}

func scopeText(scope string) string {
	switch scope {
	case "local":
		return "all metadata and tensor payloads"
	case "local_filtered":
		return "all metadata and selected tensor payloads"
	case "local_metadata":
		return "all metadata and tensor descriptors; unmatched payloads not checked"
	case "local_metadata_filtered":
		return "all metadata and selected tensor descriptors; unmatched payloads not checked"
	default:
		return scope
	}
}

func writeGroups(b *strings.Builder, name string, groups map[string]Counts) {
	if len(groups) == 0 {
		return
	}
	fmt.Fprintf(b, "\n%s:\n", name)
	for _, key := range unionKeys(groups, nil) {
		counts := groups[key]
		fmt.Fprintf(b, "  %s: %s; %s -> %s\n", terminalText(key), countSummary(counts), ollamaformat.HumanBytes(counts.LeftBytes), ollamaformat.HumanBytes(counts.RightBytes))
	}
}

func countSummary(c Counts) string {
	var parts []string
	for _, entry := range []struct {
		count int
		word  string
	}{
		{c.Equal, "unchanged"},
		{c.Changed, "changed"},
		{c.Added, "added"},
		{c.Removed, "removed"},
	} {
		if entry.count > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", entry.count, entry.word))
		}
	}
	if c.NotChecked > 0 {
		parts = append(parts, fmt.Sprintf("%d payloads not checked", c.NotChecked))
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}

func changeSummary(n int) string {
	if n == 0 {
		return "unchanged"
	}
	return fmt.Sprintf("%d %s", n, plural(n, "change", "changes"))
}

func writeMetadata(b *strings.Builder, d MetadataChange, all bool) {
	path := "metadata" + d.Path
	if d.LeftPresent {
		fmt.Fprintf(b, "\n--- %s\n", terminalText(path))
	} else {
		fmt.Fprintln(b, "\n--- /dev/null")
	}
	if d.RightPresent {
		fmt.Fprintf(b, "+++ %s\n", terminalText(path))
	} else {
		fmt.Fprintln(b, "+++ /dev/null")
	}
	fmt.Fprintf(b, "  classification: %s\n", metadataClass(d.Path))
	left, leftProse := d.Left.(proseMetadata)
	right, rightProse := d.Right.(proseMetadata)
	if (leftProse || !d.LeftPresent) && (rightProse || !d.RightPresent) && (leftProse || rightProse) {
		writeProseDiff(b, left.lines, right.lines, all)
		return
	}
	if d.LeftPresent {
		fmt.Fprintf(b, "- %s\n", displayJSON(d.Left, all))
	}
	if d.RightPresent {
		fmt.Fprintf(b, "+ %s\n", displayJSON(d.Right, all))
	}
}

func writeProseDiff(b *strings.Builder, left, right []string, all bool) {
	const (
		contextLines = 2
		defaultHunks = 3
		defaultLines = 40
	)
	type section struct {
		prefix byte
		lines  []string
	}
	groups := difflib.NewMatcherWithJunk(left, right, false, nil).GetGroupedOpCodes(contextLines)
	linesWritten := 0
	for i, group := range groups {
		if !all && (i >= defaultHunks || linesWritten >= defaultLines) {
			fmt.Fprintln(b, "... metadata diff truncated; use --all.")
			return
		}
		first, last := group[0], group[len(group)-1]
		fmt.Fprintf(b, "@@ -%s +%s @@\n", unifiedRange(first.I1, last.I2), unifiedRange(first.J1, last.J2))
		for _, operation := range group {
			var sections []section
			switch operation.Tag {
			case 'e':
				sections = append(sections, section{' ', left[operation.I1:operation.I2]})
			case 'r':
				sections = append(sections,
					section{'-', left[operation.I1:operation.I2]},
					section{'+', right[operation.J1:operation.J2]},
				)
			case 'd':
				sections = append(sections, section{'-', left[operation.I1:operation.I2]})
			case 'i':
				sections = append(sections, section{'+', right[operation.J1:operation.J2]})
			}
			for _, section := range sections {
				for _, line := range section.lines {
					if !all && linesWritten >= defaultLines {
						fmt.Fprintln(b, "... metadata diff truncated; use --all.")
						return
					}
					fmt.Fprintf(b, "%c%s\n", section.prefix, line)
					linesWritten++
				}
			}
		}
	}
}

func unifiedRange(start, end int) string {
	length := end - start
	if length == 0 {
		return fmt.Sprintf("%d,0", start)
	}
	if length == 1 {
		return strconv.Itoa(start + 1)
	}
	return fmt.Sprintf("%d,%d", start+1, length)
}

func writeTensor(b *strings.Builder, d TensorChange, all bool) {
	path := "tensor/" + d.Role + "/" + d.Name
	if d.Left != nil {
		fmt.Fprintf(b, "\n--- %s\n", terminalText(path))
	} else {
		fmt.Fprintln(b, "\n--- /dev/null")
	}
	if d.Right != nil {
		fmt.Fprintf(b, "+++ %s\n", terminalText(path))
	} else {
		fmt.Fprintln(b, "+++ /dev/null")
	}

	writeTensorField(b, "model dtype", d.Left, d.Right, modelDTypeText)
	writeTensorField(b, "stored as", d.Left, d.Right, storedTensorText)
	writeTensorField(b, "companions", d.Left, d.Right, func(t *Tensor) string { return strings.Join(t.Companions, ", ") })
	writeTensorField(b, "companion of", d.Left, d.Right, func(t *Tensor) string { return t.CompanionOf })
	if (d.Left != nil && len(d.Left.Metadata) > 0) || (d.Right != nil && len(d.Right.Metadata) > 0) {
		writeTensorField(b, "header metadata", d.Left, d.Right, func(t *Tensor) string { return displayJSON(t.Metadata, all) })
	}
	if (d.Left != nil && d.Left.Blob != "") || (d.Right != nil && d.Right.Blob != "") {
		writeTensorField(b, "blob", d.Left, d.Right, func(t *Tensor) string { return t.Blob })
	}
	writePayload(b, d)
	if all {
		writeTensorField(b, "location", d.Left, d.Right, tensorLocation)
	}
}

func writeTensorField(b *strings.Builder, label string, left, right *Tensor, value func(*Tensor) string) {
	leftValue, rightValue := "", ""
	if left != nil {
		leftValue = value(left)
	}
	if right != nil {
		rightValue = value(right)
	}
	if leftValue == "" && rightValue == "" {
		return
	}
	if left != nil && right != nil && leftValue == rightValue {
		fmt.Fprintf(b, "  %s: %s\n", label, leftValue)
		return
	}
	if left != nil && leftValue != "" {
		fmt.Fprintf(b, "- %s: %s\n", label, leftValue)
	}
	if right != nil && rightValue != "" {
		fmt.Fprintf(b, "+ %s: %s\n", label, rightValue)
	}
}

func writePayload(b *strings.Builder, d TensorChange) {
	switch d.Verification {
	case "blob":
		fmt.Fprintf(b, "  payload: identical (shared blob %s)\n", d.Left.Blob)
	case "same_file":
		fmt.Fprintln(b, "  payload: identical (same file range)")
	case "sha256":
		left, right := "", ""
		if d.Left != nil {
			left = d.Left.SHA256
		}
		if d.Right != nil {
			right = d.Right.SHA256
		}
		if left != "" && left == right {
			fmt.Fprintf(b, "  payload sha256: %s (identical)\n", left)
			return
		}
		if left != "" {
			fmt.Fprintf(b, "- payload sha256: %s\n", left)
		}
		if right != "" {
			fmt.Fprintf(b, "+ payload sha256: %s\n", right)
		}
	case "not_checked":
		fmt.Fprintln(b, "! payload not checked")
	}
}

func modelDTypeText(t *Tensor) string {
	if t.Quantization == nil {
		return t.ModelDType
	}
	return fmt.Sprintf("%s (%d-bit, group %d; logical shape %s)", t.ModelDType, t.Quantization.Bits, t.Quantization.GroupSize, shapeText(t.Quantization.LogicalShape))
}

func storedTensorText(t *Tensor) string {
	return fmt.Sprintf("%s %s; %d elements; %s; %s %s-endian", t.DType, shapeText(t.Shape), t.Elements, ollamaformat.HumanBytes(t.Bytes), t.Format, t.ByteOrder)
}

func tensorLocation(t *Tensor) string {
	parts := []string{fmt.Sprintf("%s @%d", terminalText(t.File), t.Offset)}
	if t.Layer != "" {
		parts = append(parts, "layer "+terminalText(t.Layer))
	}
	return strings.Join(parts, "; ")
}

func shapeText(shape []uint64) string {
	data, _ := json.Marshal(shape)
	return string(data)
}

func shortHash(digest string) string {
	const prefix = "sha256:"
	digest = strings.TrimPrefix(digest, prefix)
	if len(digest) > 12 {
		digest = digest[:12] + "…"
	}
	return prefix + digest
}

func terminalText(s string) string {
	quoted := strconv.QuoteToGraphic(s)
	return quoted[1 : len(quoted)-1]
}

func plural(n int, singular, plural string) string {
	if n == 1 {
		return singular
	}
	return plural
}

func displayJSON(v any, all bool) string {
	if prose, ok := v.(proseMetadata); ok {
		v = prose.text
	}
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("<%s>", err)
	}
	if !all && len(data) > 300 {
		return string(data[:300]) + "… (full value in --all)"
	}
	return string(data)
}

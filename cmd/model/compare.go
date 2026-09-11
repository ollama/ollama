package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"maps"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strings"
)

type Options struct {
	LeftStore    string
	RightStore   string
	MetadataOnly bool
	Stats        bool
	Tensor       string // regular expression on the original tensor name
}

type Source struct {
	Reference string
	Path      string
	Store     string
	Format    string
	Digest    string
	Artifacts []Artifact
}

type Tensor struct {
	Name         string
	Role         string
	ModelDType   string
	DType        string
	Shape        []uint64
	Elements     uint64
	Bytes        int64
	Format       string
	MediaType    string
	ByteOrder    string
	File         string
	Layer        string
	Blob         string
	Offset       int64
	SHA256       string
	Metadata     map[string]string
	Quantization *Quantization
	Companions   []string
	CompanionOf  string
}

type Quantization struct {
	Type         string
	Bits         int
	GroupSize    int
	LogicalShape []uint64
}

type TensorChange struct {
	Name         string
	Role         string
	Status       string // equal, changed, added, removed, not_checked
	Changes      []string
	Verification string // blob, same_file, sha256, not_checked
	Layout       bool
	Left         *Tensor
	Right        *Tensor
}

type Counts struct {
	Total      int
	Equal      int
	Changed    int
	Added      int
	Removed    int
	NotChecked int
	LeftBytes  int64
	RightBytes int64
}

type Summary struct {
	Counts
	MetadataChanges    int
	SemanticMetadata   int
	ProvenanceMetadata int
	DescriptorChanges  int
	DTypeTransitions   int
	LayoutChanges      int
	BlobMatched        int
	SameFileMatched    int
	PayloadHashed      int
	PayloadUnchecked   int
	BytesHashed        int64
	LeftBlobs          int
	RightBlobs         int
	SharedBlobs        int
	LeftFiles          int
	RightFiles         int
	Components         map[string]Counts
	ModelDTypes        map[string]Counts
	StorageDTypes      map[string]Counts
	PayloadOnlyDTypes  map[string]int
}

type Report struct {
	Complete        bool // all inputs checked in the explicitly requested scope
	PayloadComplete bool
	Left            Source
	Right           Source
	Scope           string
	Filter          string
	Equal           bool
	Summary         Summary
	Metadata        []MetadataChange
	Tensors         []TensorChange
	Renames         []TensorRename
	ExpertFusions   []ExpertFusion
	Stats           *StatsReport
	Warnings        []string
}

type inventory struct {
	source     Source
	metadata   map[string]any
	tensors    map[string]*Tensor
	files      map[string]os.FileInfo
	mediaTypes map[string]any
}

type preparedComparison struct {
	left, right *inventory
	opts        Options
	filter      *regexp.Regexp
}

// Compare resolves each input once. Equal means equal in the stated scope, not
// functional model equivalence. Content-addressed store digests are trusted, not audited.
func Compare(ctx context.Context, left, right string, opts Options) (*Report, error) {
	p, err := prepareComparison(ctx, left, right, opts)
	if err != nil {
		return nil, err
	}
	return p.compare(ctx)
}

func prepareComparison(ctx context.Context, left, right string, opts Options) (*preparedComparison, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var filter *regexp.Regexp
	if opts.Tensor != "" {
		var err error
		filter, err = regexp.Compile(opts.Tensor)
		if err != nil {
			return nil, fmt.Errorf("tensor filter: %w", err)
		}
	}
	a, err := inspect(ctx, left, opts.LeftStore)
	if err != nil {
		return nil, fmt.Errorf("left: %w", err)
	}
	b, err := inspect(ctx, right, opts.RightStore)
	if err != nil {
		return nil, fmt.Errorf("right: %w", err)
	}
	return &preparedComparison{left: a, right: b, opts: opts, filter: filter}, nil
}

func (p *preparedComparison) compare(ctx context.Context) (*Report, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	a, b, opts, filter := p.left, p.right, p.opts, p.filter
	r := &Report{
		Left: a.source, Right: b.source, Scope: "local", Filter: opts.Tensor,
		Metadata: []MetadataChange{}, Tensors: []TensorChange{},
		Warnings: []string{"Blob digests are trusted; this is not a store integrity audit."},
	}
	if opts.Stats {
		r.Warnings = append(r.Warnings, "NMSE is diagnostic and does not change encoded equality or exit status.")
	} else {
		r.Warnings = append(r.Warnings, "Payload equality compares encoded bytes, not numerical equivalence.")
	}
	if opts.MetadataOnly {
		r.Scope = "local_metadata"
	}
	if filter != nil {
		r.Scope += "_filtered"
	}
	metadataDiff("", a.metadata, b.metadata, true, true, &r.Metadata)
	keys := unionKeys(a.tensors, b.tensors)
	selected := make(map[string]bool)
	for _, k := range keys {
		x := a.tensors[k]
		if x == nil {
			x = b.tensors[k]
		}
		if filter == nil || filter.MatchString(x.Name) {
			selected[k] = true
			// Selecting a packed weight must also check its numeric companions.
			for _, inv := range []*inventory{a, b} {
				if t := inv.tensors[k]; t != nil {
					for _, c := range t.Companions {
						selected[tensorKey(t.Role, c)] = true
					}
				}
			}
		}
	}
	if filter != nil && len(selected) == 0 {
		return nil, fmt.Errorf("tensor filter %q matched no tensors", opts.Tensor)
	}
	h := payloadHasher{cache: make(map[payloadRange]string), buffer: make([]byte, 1<<20)}
	if opts.Stats {
		h.stats = &StatsReport{}
		h.statsScanned = make(map[*Tensor]bool)
		h.statsData = make(map[string][]byte)
	}
	for _, k := range keys {
		if !selected[k] {
			continue
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		leftTensor, rightTensor := a.tensors[k], b.tensors[k]
		if h.stats != nil && shouldCompareNumerically(leftTensor, rightTensor) {
			comparison, err := h.compareNumerically(ctx, leftTensor, rightTensor, a, b)
			if err != nil {
				return nil, fmt.Errorf("tensor statistics %q: %w", k, err)
			}
			if comparison != nil {
				h.stats.Comparisons = append(h.stats.Comparisons, *comparison)
			}
		}
		d, err := compareTensor(ctx, leftTensor, rightTensor, opts.MetadataOnly, &h)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", k, err)
		}
		r.Tensors = append(r.Tensors, d)
	}
	r.ExpertFusions = detectExpertFusions(r.Tensors)
	r.Renames = detectRenames(r.Tensors, r.ExpertFusions)
	if h.stats != nil {
		if err := h.finishStats(ctx, r.Tensors, a, b); err != nil {
			return nil, err
		}
		r.Stats = h.stats
	}
	for _, inv := range []*inventory{a, b} {
		if err := inv.checkFiles(); err != nil {
			return nil, err
		}
	}
	r.summarize()
	r.Summary.BytesHashed = h.bytes
	return r, nil
}

func tensorKey(role, name string) string { return role + "/" + name }

func compareTensor(ctx context.Context, a, b *Tensor, metadataOnly bool, h *payloadHasher) (TensorChange, error) {
	t := a
	if t == nil {
		t = b
	}
	d := TensorChange{Name: t.Name, Role: t.Role, Left: a, Right: b, Status: "equal", Verification: "not_checked", Changes: []string{}}
	if a == nil {
		d.Status, d.Changes = "added", []string{"added"}
	} else if b == nil {
		d.Status, d.Changes = "removed", []string{"removed"}
	} else {
		for _, field := range []struct {
			name  string
			equal bool
		}{
			{"model_dtype", a.ModelDType == b.ModelDType},
			{"dtype", a.DType == b.DType},
			{"shape", slices.Equal(a.Shape, b.Shape)},
			{"bytes", a.Bytes == b.Bytes},
			{"format", a.Format == b.Format},
			{"media_type", a.MediaType == b.MediaType},
			{"byte_order", a.ByteOrder == b.ByteOrder},
			{"metadata", maps.Equal(a.Metadata, b.Metadata)},
			{"quantization", reflect.DeepEqual(a.Quantization, b.Quantization)},
			{"companions", slices.Equal(a.Companions, b.Companions)},
			{"companion_of", a.CompanionOf == b.CompanionOf},
		} {
			if !field.equal {
				d.Changes = append(d.Changes, field.name)
			}
		}
		d.Layout = a.Blob != b.Blob || a.Offset != b.Offset || a.Layer != b.Layer
		if a.Blob != "" && a.Blob == b.Blob {
			if a.Offset != b.Offset || a.Bytes != b.Bytes || a.DType != b.DType || a.ByteOrder != b.ByteOrder || !slices.Equal(a.Shape, b.Shape) || !maps.Equal(a.Metadata, b.Metadata) {
				return d, fmt.Errorf("matching blob digests have conflicting tensor descriptors")
			}
			d.Verification = "blob"
		} else if a.File == b.File && a.Offset == b.Offset && a.Bytes == b.Bytes {
			d.Verification = "same_file"
		}
	}
	if d.Verification == "not_checked" && !metadataOnly {
		for _, t := range []*Tensor{a, b} {
			if t != nil {
				sum, err := h.hash(ctx, t.File, t.Offset, t.Bytes)
				if err != nil {
					return d, err
				}
				t.SHA256 = sum
			}
		}
		d.Verification = "sha256"
		if a != nil && b != nil && a.SHA256 != b.SHA256 {
			d.Changes = append(d.Changes, "payload")
		}
	}
	if a != nil && b != nil {
		if len(d.Changes) > 0 {
			d.Status = "changed"
		} else if d.Verification == "not_checked" {
			d.Status = "not_checked"
		}
	}
	return d, nil
}

type payloadRange struct {
	path         string
	offset, size int64
}

type payloadHasher struct {
	cache        map[payloadRange]string
	buffer       []byte
	bytes        int64
	stats        *StatsReport
	statsScanned map[*Tensor]bool
	statsData    map[string][]byte
}

func (h *payloadHasher) hash(ctx context.Context, path string, offset, size int64) (string, error) {
	key := payloadRange{path, offset, size}
	if sum, ok := h.cache[key]; ok {
		return sum, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	r := io.NewSectionReader(f, offset, size)
	hash := sha256.New()
	for remaining := size; remaining > 0; {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := io.ReadFull(r, h.buffer[:min(int64(len(h.buffer)), remaining)])
		if err != nil {
			return "", err
		}
		hash.Write(h.buffer[:n])
		h.bytes += int64(n)
		remaining -= int64(n)
	}
	sum := hex.EncodeToString(hash.Sum(nil))
	h.cache[key] = sum
	return sum, nil
}

func (r *Report) summarize() {
	r.Complete = true
	r.Summary = Summary{
		MetadataChanges: len(r.Metadata),
		Components:      make(map[string]Counts), ModelDTypes: make(map[string]Counts), StorageDTypes: make(map[string]Counts), PayloadOnlyDTypes: make(map[string]int),
	}
	for _, d := range r.Metadata {
		if metadataClass(d.Path) == "provenance" {
			r.Summary.ProvenanceMetadata++
		} else {
			r.Summary.SemanticMetadata++
		}
	}
	leftBlobs, rightBlobs := make(map[string]bool), make(map[string]bool)
	leftFiles, rightFiles := make(map[string]bool), make(map[string]bool)
	for _, d := range r.Tensors {
		r.Summary.Counts.add(d)
		t := d.Right
		if t == nil {
			t = d.Left
		}
		component := tensorComponent(t.Role, t.Name)
		c := r.Summary.Components[component]
		c.add(d)
		r.Summary.Components[component] = c
		modelDType := transition(d.Left, d.Right, func(t *Tensor) string { return t.ModelDType })
		c = r.Summary.ModelDTypes[modelDType]
		c.add(d)
		r.Summary.ModelDTypes[modelDType] = c
		storageDType := transition(d.Left, d.Right, func(t *Tensor) string { return t.DType })
		c = r.Summary.StorageDTypes[storageDType]
		c.add(d)
		r.Summary.StorageDTypes[storageDType] = c
		for _, side := range []struct {
			t            *Tensor
			blobs, files map[string]bool
		}{{d.Left, leftBlobs, leftFiles}, {d.Right, rightBlobs, rightFiles}} {
			if side.t == nil {
				continue
			}
			if side.t.Blob != "" {
				side.blobs[side.t.Blob] = true
			} else {
				side.files[side.t.File] = true
			}
		}
		if d.Layout {
			r.Summary.LayoutChanges++
		}
		if d.Status == "added" || d.Status == "removed" {
			r.Summary.DescriptorChanges++
		} else if d.Status == "changed" {
			payloadOnly := slices.Equal(d.Changes, []string{"payload"})
			if payloadOnly {
				r.Summary.PayloadOnlyDTypes[t.ModelDType]++
			} else {
				r.Summary.DescriptorChanges++
			}
			if d.Left != nil && d.Right != nil && (d.Left.ModelDType != d.Right.ModelDType || d.Left.DType != d.Right.DType) {
				r.Summary.DTypeTransitions++
			}
		}
		switch d.Verification {
		case "blob":
			r.Summary.BlobMatched++
		case "same_file":
			r.Summary.SameFileMatched++
		case "sha256":
			r.Summary.PayloadHashed++
		case "not_checked":
			r.Summary.PayloadUnchecked++
		}
	}
	r.Summary.LeftBlobs, r.Summary.RightBlobs = len(leftBlobs), len(rightBlobs)
	r.Summary.LeftFiles, r.Summary.RightFiles = len(leftFiles), len(rightFiles)
	for blob := range leftBlobs {
		if rightBlobs[blob] {
			r.Summary.SharedBlobs++
		}
	}
	r.PayloadComplete = len(r.Tensors) > 0 && r.Summary.PayloadUnchecked == 0
	r.Equal = len(r.Metadata) == 0 && r.Summary.Changed+r.Summary.Added+r.Summary.Removed == 0
}

func transition(a, b *Tensor, value func(*Tensor) string) string {
	if a == nil {
		return value(b)
	}
	if b == nil {
		return value(a)
	}
	if av, bv := value(a), value(b); av != bv {
		return av + " -> " + bv
	}
	return value(a)
}

func metadataClass(path string) string {
	path = strings.ToLower(path)
	for _, prefix := range []string{"/manifest/", "/manifest_config/rootfs", "/manifest_config/history", "/manifest_config/created", "/license"} {
		if path == strings.TrimSuffix(prefix, "/") || strings.HasPrefix(path, prefix) {
			return "provenance"
		}
	}
	return "semantic"
}

func tensorComponent(role, name string) string {
	if role != "model" {
		return role
	}
	for _, candidate := range []struct {
		name     string
		prefixes []string
	}{
		{"draft", []string{"draft.", "mtp."}},
		{"vision", []string{"model.vision_tower.", "model.vision_embedder.", "model.embed_vision.", "model.vision_projection.", "vision_model.", "vision_tower.", "vision_projection.", "visual."}},
		{"audio", []string{"model.audio_tower.", "model.embed_audio.", "audio_model.", "audio_tower.", "sound_encoder.", "sound_projection."}},
		{"text", []string{"model.language_model.", "language_model."}},
	} {
		for _, prefix := range candidate.prefixes {
			if strings.HasPrefix(name, prefix) {
				return candidate.name
			}
		}
	}
	return "model"
}

func (c *Counts) add(d TensorChange) {
	c.Total++
	switch d.Status {
	case "equal":
		c.Equal++
	case "changed":
		c.Changed++
	case "added":
		c.Added++
	case "removed":
		c.Removed++
	case "not_checked":
		c.NotChecked++
	}
	if d.Left != nil {
		c.LeftBytes += d.Left.Bytes
	}
	if d.Right != nil {
		c.RightBytes += d.Right.Bytes
	}
}

package main

import (
	"context"
	"fmt"
	"io/fs"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

func (inv *inventory) directory(ctx context.Context, root string) error {
	var indexes []string
	var tensorFiles []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if path != root && strings.HasPrefix(entry.Name(), ".") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		fi, err := inv.track(path)
		if err != nil {
			return err
		}
		if fi.IsDir() {
			if !entry.IsDir() {
				return fmt.Errorf("directory symlink is not supported: %s", path)
			}
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		a := Artifact{Name: filepath.ToSlash(rel), File: path, Bytes: fi.Size()}
		inv.source.Artifacts = append(inv.source.Artifacts, a)
		switch {
		case strings.HasSuffix(path, ".safetensors.index.json"):
			indexes = append(indexes, path)
		case strings.HasSuffix(path, ".safetensors"), strings.HasSuffix(path, ".gguf"):
			tensorFiles = append(tensorFiles, path)
			role := "model"
			if strings.HasPrefix(a.Name, "draft/") || strings.HasPrefix(a.Name, "mtp/") {
				role = "draft"
			}
			if strings.HasPrefix(entry.Name(), "mmproj") {
				role = "projector"
			}
			return inv.tensorFile(ctx, a, role)
		default:
			return inv.artifact(ctx, a.Name, a, "")
		}
		return nil
	})
	if err != nil {
		return err
	}
	// Index-less HF exports still advertise a shard count in their filenames.
	// Check it rather than accepting a directory containing only the first shard.
	shardName := regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.safetensors$`)
	files := make(map[string]bool)
	for _, file := range tensorFiles {
		files[file] = true
	}
	checked := make(map[string]bool)
	for _, file := range tensorFiles {
		parts := shardName.FindStringSubmatch(filepath.Base(file))
		if parts == nil {
			continue
		}
		n, _ := strconv.Atoi(parts[3])
		part, _ := strconv.Atoi(parts[2])
		if n < 1 || part < 1 || part > n {
			return fmt.Errorf("invalid shard number: %s", file)
		}
		key := filepath.Join(filepath.Dir(file), parts[1]+"-of-"+parts[3])
		if checked[key] {
			continue
		}
		checked[key] = true
		for i := 1; i <= n; i++ {
			expected := filepath.Join(filepath.Dir(file), fmt.Sprintf("%s-%05d-of-%05d.safetensors", parts[1], i, n))
			if !files[expected] {
				return fmt.Errorf("missing safetensors shard: %s", expected)
			}
		}
	}
	for _, index := range indexes {
		if err := inv.validateIndex(index); err != nil {
			return err
		}
	}
	// A monolithic model beside shard files is usually a stale conversion, not
	// an instruction to concatenate both checkpoints.
	for _, path := range tensorFiles {
		if filepath.Base(path) != "model.safetensors" {
			continue
		}
		for _, other := range tensorFiles {
			if filepath.Dir(other) == filepath.Dir(path) && strings.HasPrefix(filepath.Base(other), "model-") && strings.HasSuffix(other, ".safetensors") {
				return fmt.Errorf("ambiguous monolithic model and shards in %s", filepath.Dir(path))
			}
		}
	}
	return nil
}

func (inv *inventory) validateIndex(path string) error {
	data, err := inv.readMetadata(path)
	if err != nil {
		return err
	}
	v, err := decodeJSON(data)
	if err != nil {
		return err
	}
	index, ok := v.(map[string]any)
	if !ok {
		return fmt.Errorf("invalid safetensors index %s", path)
	}
	weights, ok := index["weight_map"].(map[string]any)
	if !ok || len(weights) == 0 {
		return fmt.Errorf("empty or missing weight_map in %s", path)
	}
	actual := make(map[string]map[string]bool)
	for _, t := range inv.tensors {
		if t.Format != "safetensors" {
			continue
		}
		if actual[t.File] == nil {
			actual[t.File] = make(map[string]bool)
		}
		actual[t.File][t.Name] = true
	}
	used := make(map[string]bool)
	for _, name := range unionKeys(weights, nil) {
		shard, ok := weights[name].(string)
		if !ok || !filepath.IsLocal(shard) {
			return fmt.Errorf("invalid shard path for %q in %s", name, path)
		}
		file := filepath.Join(filepath.Dir(path), shard)
		if !actual[file][name] {
			return fmt.Errorf("index %s: tensor %q missing from shard %s", path, name, shard)
		}
		used[file] = true
	}
	for file := range used {
		for name := range actual[file] {
			shard, ok := weights[name].(string)
			if !ok || filepath.Join(filepath.Dir(path), shard) != file {
				return fmt.Errorf("unindexed tensor %q in %s", name, file)
			}
		}
	}
	for file := range actual {
		if filepath.Dir(file) == filepath.Dir(path) && !used[file] {
			return fmt.Errorf("unindexed safetensors file beside %s: %s", path, file)
		}
	}
	// Shard filenames and total_size are physical layout, not semantic config.
	delete(index, "weight_map")
	if meta, ok := index["metadata"].(map[string]any); ok {
		delete(meta, "total_size")
		if len(meta) == 0 {
			delete(index, "metadata")
		}
	}
	if len(index) == 0 {
		return nil
	}
	rel, err := filepath.Rel(inv.source.Path, filepath.Dir(path))
	if err != nil {
		return err
	}
	key := "safetensors_index/" + filepath.ToSlash(rel)
	if _, ok := inv.metadata[key]; ok {
		return fmt.Errorf("ambiguous indexes in %s", filepath.Dir(path))
	}
	inv.metadata[key] = index
	return nil
}

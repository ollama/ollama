package cache

import (
	"fmt"
	"strconv"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const snapshotFormatVersion = "1"

// PersistedSnapshots is a restorable set of cache snapshots stored on disk.
type PersistedSnapshots struct {
	Offset    int
	Snapshots []Snapshot
}

// SaveSnapshots persists a restorable snapshot set to a safetensors file.
func SaveSnapshots(path string, offset int, snapshots []Snapshot) error {
	arrays := make(map[string]*mlx.Array)
	metadata := map[string]string{
		"ollama.kvconnector.version": snapshotFormatVersion,
		"ollama.kvconnector.offset":  strconv.Itoa(offset),
		"ollama.kvconnector.count":   strconv.Itoa(len(snapshots)),
	}

	var state []*mlx.Array
	for i, snap := range snapshots {
		prefix := fmt.Sprintf("snapshot.%d", i)
		switch s := snap.(type) {
		case nil:
			metadata[prefix+".kind"] = "nil"
		case *kvSnapshot:
			s.copyOut()
			if s.keys == nil || s.values == nil {
				return fmt.Errorf("snapshot %d: kv snapshot has no data", i)
			}
			metadata[prefix+".kind"] = "kv"
			metadata[prefix+".from"] = strconv.Itoa(s.fromOffset)
			metadata[prefix+".to"] = strconv.Itoa(s.toOffset)
			arrays[prefix+".keys"] = s.keys
			arrays[prefix+".values"] = s.values
			state = append(state, s.keys, s.values)
		case *rotatingSnapshot:
			s.copyOut()
			if s.keys == nil || s.values == nil {
				return fmt.Errorf("snapshot %d: rotating snapshot has no data", i)
			}
			metadata[prefix+".kind"] = "rotating"
			metadata[prefix+".from"] = strconv.Itoa(s.fromOffset)
			metadata[prefix+".to"] = strconv.Itoa(s.toOffset)
			metadata[prefix+".idx"] = strconv.Itoa(s.idx)
			arrays[prefix+".keys"] = s.keys
			arrays[prefix+".values"] = s.values
			state = append(state, s.keys, s.values)
		case *recurrentSnapshot:
			if s.convState == nil || s.deltaState == nil {
				return fmt.Errorf("snapshot %d: recurrent snapshot has no data", i)
			}
			metadata[prefix+".kind"] = "recurrent"
			metadata[prefix+".offset"] = strconv.Itoa(s.offset)
			arrays[prefix+".conv"] = s.convState
			arrays[prefix+".delta"] = s.deltaState
			state = append(state, s.convState, s.deltaState)
		default:
			return fmt.Errorf("snapshot %d: unsupported snapshot type %T", i, snap)
		}
	}

	if len(state) > 0 {
		mlx.Eval(state...)
	}

	return mlx.SaveSafetensorsWithMetadata(path, arrays, metadata)
}

// LoadSnapshots restores a snapshot set from a safetensors file.
func LoadSnapshots(path string) (_ *PersistedSnapshots, err error) {
	sf, err := mlx.LoadSafetensorsNative(path)
	if err != nil {
		return nil, err
	}
	defer sf.Free()

	if got := sf.GetMetadata("ollama.kvconnector.version"); got != snapshotFormatVersion {
		return nil, fmt.Errorf("unsupported kvconnector snapshot version %q", got)
	}

	offset, err := parseSnapshotMetadataInt(sf, "ollama.kvconnector.offset")
	if err != nil {
		return nil, err
	}
	count, err := parseSnapshotMetadataInt(sf, "ollama.kvconnector.count")
	if err != nil {
		return nil, err
	}

	out := &PersistedSnapshots{
		Offset:    offset,
		Snapshots: make([]Snapshot, count),
	}
	defer func() {
		if err != nil {
			for _, snap := range out.Snapshots {
				if snap != nil {
					snap.Close()
				}
			}
		}
	}()

	for i := range count {
		prefix := fmt.Sprintf("snapshot.%d", i)
		switch kind := sf.GetMetadata(prefix + ".kind"); kind {
		case "", "nil":
			continue
		case "kv":
			from, err := parseSnapshotMetadataInt(sf, prefix+".from")
			if err != nil {
				return nil, err
			}
			to, err := parseSnapshotMetadataInt(sf, prefix+".to")
			if err != nil {
				return nil, err
			}
			keys := sf.Get(prefix + ".keys")
			values := sf.Get(prefix + ".values")
			if keys == nil || values == nil {
				return nil, fmt.Errorf("snapshot %d: missing kv arrays", i)
			}
			mlx.Pin(keys, values)
			out.Snapshots[i] = &kvSnapshot{keys: keys, values: values, fromOffset: from, toOffset: to}
		case "rotating":
			from, err := parseSnapshotMetadataInt(sf, prefix+".from")
			if err != nil {
				return nil, err
			}
			to, err := parseSnapshotMetadataInt(sf, prefix+".to")
			if err != nil {
				return nil, err
			}
			idx, err := parseSnapshotMetadataInt(sf, prefix+".idx")
			if err != nil {
				return nil, err
			}
			keys := sf.Get(prefix + ".keys")
			values := sf.Get(prefix + ".values")
			if keys == nil || values == nil {
				return nil, fmt.Errorf("snapshot %d: missing rotating arrays", i)
			}
			mlx.Pin(keys, values)
			out.Snapshots[i] = &rotatingSnapshot{keys: keys, values: values, fromOffset: from, toOffset: to, idx: idx}
		case "recurrent":
			snapOffset, err := parseSnapshotMetadataInt(sf, prefix+".offset")
			if err != nil {
				return nil, err
			}
			conv := sf.Get(prefix + ".conv")
			delta := sf.Get(prefix + ".delta")
			if conv == nil || delta == nil {
				return nil, fmt.Errorf("snapshot %d: missing recurrent arrays", i)
			}
			mlx.Pin(conv, delta)
			out.Snapshots[i] = &recurrentSnapshot{convState: conv, deltaState: delta, offset: snapOffset}
		default:
			return nil, fmt.Errorf("snapshot %d: unsupported snapshot kind %q", i, kind)
		}
	}

	return out, nil
}

func parseSnapshotMetadataInt(sf *mlx.SafetensorsFile, key string) (int, error) {
	raw := sf.GetMetadata(key)
	if raw == "" {
		return 0, fmt.Errorf("missing kvconnector metadata %q", key)
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid kvconnector metadata %q: %w", key, err)
	}
	return v, nil
}

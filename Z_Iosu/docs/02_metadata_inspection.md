# GGUF Vision Metadata Inspection Guide

## Purpose
Determine whether a model file contains the vision-related keys required for Ollama to expose the `vision` capability.

## Key Vision Metadata Keys
| Category | Key Pattern | Description |
| -------- | ----------- | ----------- |
| Presence Gate | `<arch>.vision.block_count` | Number of vision transformer / encoder blocks (primary gate) |
| Geometry | `<arch>.vision.image_size` | Input image size (square) |
| Geometry | `<arch>.vision.patch_size` | Patch size used for tokenization |
| Channels | `<arch>.vision.num_channels` | Input channels (usually 3) |
| Attention | `<arch>.vision.attention.head_count` | Vision attention heads |
| Embedding | `<arch>.vision.embedding_length` | Embedding dimension for patches |
| Tiles (mllama) | `<arch>.vision.max_num_tiles` | Max tiles supported (multi-tile) |
| Misc (qwen25vl) | `<arch>.vision.max_pixels` | Upper bound on pixels processed |

`<arch>` examples: `llama4`, `mllama`, `gemma3`, `qwen25vl`, `mistral3`.

## Methods
### 1. Using `strings` (Quick Heuristic)
```bash
strings model.gguf | grep -i vision.block_count
```
If no output → either key absent or file compressed / `strings` not present (move to method 2).

### 2. Minimal Go Inspector (Recommended)
Create `inspect.go`:
```go
package main
import (
  "fmt"
  "os"
  "github.com/ollama/ollama/fs/gguf"
)
func main(){
  if len(os.Args)<2 { fmt.Println("usage: inspect <model.gguf>"); return }
  f, err := gguf.Open(os.Args[1])
  if err!=nil { panic(err) }
  defer f.Close()
  // Iterate over all key-values lazily
  // We don't have direct iteration helper here, so probe expected keys
  arches := []string{"llama4","mllama","gemma3","gemma3n","mistral3","qwen25vl"}
  for _, a := range arches {
    k := a+".vision.block_count"
    if f.KeyValue(k).Valid() { fmt.Printf("FOUND %s = %v\n", k, f.KeyValue(k).Value) }
  }
}
```
Build & run:
```bash
go mod init tmpinspect
go get github.com/ollama/ollama@<commit-or-tag>
go build -o inspect inspect.go
./inspect model.gguf
```

### 3. Programmatic Enumeration (Advanced)
Extend the snippet to walk the full key set by forcing lazy loaders if needed (modifying library not shown here). For deep dives, consider adding instrumentation in `fs/gguf/gguf.go`.

## Validating Capability Chain
1. Key exists → `Capabilities()` can add `vision`.
2. Absent key but projector present (separate GGUF loaded) → still adds vision.
3. Both missing → no `vision` capability.

## Troubleshooting Scenarios
| Scenario | Likely Cause | Action |
| -------- | ----------- | ------ |
| No `vision.block_count`, no projector | Conversion omission | Reconvert / re-pull |
| Projector file missing | Incomplete model bundle | Re-pull model, check network/cache |
| Key present but capability missing | Code path not opening file / open error | Check logs for `couldn't open model file` |
| Key present; request fails | Client payload malformed | Validate JSON structure for images |

## Logging Assistance
Temporary patch (remove after):
```go
// in server/images.go inside Capabilities after opening file
slog.Debug("vision.meta", "model", m.Name, "vision_block_present", f.KeyValue("vision.block_count").Valid())
```

## Output Checklist
| Item | Recorded? |
| ---- | --------- |
| Model path & digest |  |
| Architecture |  |
| vision.block_count value |  |
| Other vision.* key values |  |
| Projector paths list |  |

## Next Step
If metadata absent → proceed to bisect or reconvert before diffing runtime code.

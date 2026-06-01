package server

import (
	"bufio"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
)

// findLlamaQuantize locates the llama-quantize binary (installed alongside llama-server).
func findLlamaQuantize() (string, error) {
	return llm.FindLlamaCppBinary("llama-quantize")
}

// progressRegex matches llama-quantize output lines like "[ 42/ 200]"
var progressRegex = regexp.MustCompile(`\[\s*(\d+)/\s*(\d+)\]`)

const llamaCppCompatEnv = "OLLAMA_LLAMA_CPP_COMPAT"

var runLlamaQuantize = runLlamaQuantizeCommand

// quantize re-quantizes a GGUF model by shelling out to llama-quantize.
// Embedded compatibility tensors are restored afterward because llama.cpp's
// text model loader intentionally does not claim those tensors.
func quantize(in, out *os.File, orig *fsggml.GGML, newFileType fsggml.FileType, progressFn func(n uint64)) error {
	typeName := newFileType.String()
	if typeName == "" {
		return fmt.Errorf("unsupported quantization type: %v", newFileType)
	}
	if err := runLlamaQuantize(in, out, orig, newFileType, typeName, progressFn); err != nil {
		return err
	}
	if hasEmbeddedCompatibilityTensors(orig) {
		if err := restoreEmbeddedCompatibilityTensors(in, out, orig, newFileType); err != nil {
			return fmt.Errorf("failed to restore embedded compatibility tensors: %w", err)
		}
	}
	return nil
}

func copyGGUFWithLlamaQuantize(in, out *os.File, orig *fsggml.GGML, progressFn func(n uint64)) error {
	return runLlamaQuantize(in, out, orig, orig.KV().FileType(), "COPY", progressFn)
}

func needsDefaultLlavaProjectorType(ggml *fsggml.GGML) bool {
	kv := ggml.KV()
	if kv.Architecture() != "clip" || !kv.Bool("has_vision_encoder") {
		return false
	}
	if _, ok := kv["clip.projector_type"]; ok {
		return false
	}
	if _, ok := kv["clip.vision.projector_type"]; ok {
		return false
	}
	return true
}

func addDefaultLlavaProjectorType(layer *layerGGML) (*layerGGML, error) {
	blob, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}
	fp, err := os.Open(blob)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	temp, err := os.CreateTemp(filepath.Dir(blob), "projector-metadata")
	if err != nil {
		return nil, err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	kv := maps.Clone(layer.GGML.KV())
	kv["clip.projector_type"] = "mlp"

	tensors := make([]*fsggml.Tensor, 0, len(layer.GGML.Tensors().Items()))
	for _, tensor := range layer.GGML.Tensors().Items() {
		tensors = append(tensors, tensorFromFile(fp, layer.GGML.Tensors().Offset+tensor.Offset, tensor))
	}

	if err := fsggml.WriteGGUF(temp, kv, tensors); err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	newLayer, err := manifest.NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	f, err := fsggml.Decode(temp, 1024)
	if err != nil {
		return nil, err
	}

	return &layerGGML{Layer: newLayer, GGML: f}, nil
}

func runLlamaQuantizeCommand(in, out *os.File, orig *fsggml.GGML, newFileType fsggml.FileType, typeName string, progressFn func(n uint64)) error {
	quantizeExe, err := findLlamaQuantize()
	if err != nil {
		return fmt.Errorf("llama-quantize unavailable: %w", err)
	}

	slog.Info("quantizing model", "type", typeName, "input", in.Name(), "output", out.Name())

	args := llamaQuantizeArgs(orig.KV().Architecture(), newFileType, in.Name(), out.Name(), typeName)
	cmd := exec.Command(quantizeExe, args...)
	cmd.Env = llamaQuantizeEnv(os.Environ(), hasEmbeddedCompatibilityTensors(orig))

	// Parse progress from stdout
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-quantize: %w", err)
	}

	// Track total tensor size for progress reporting
	totalSize := uint64(0)
	for _, t := range orig.Tensors().Items() {
		totalSize += t.Size()
	}

	var lastReported uint64
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		if matches := progressRegex.FindStringSubmatch(line); len(matches) == 3 {
			current, _ := strconv.ParseUint(matches[1], 10, 64)
			total, _ := strconv.ParseUint(matches[2], 10, 64)
			if total > 0 && progressFn != nil {
				// progressFn expects incremental byte deltas
				done := totalSize * current / total
				if done > lastReported {
					progressFn(done - lastReported)
					lastReported = done
				}
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("llama-quantize failed: %w", err)
	}

	return nil
}

func disableLlamaCppCompat(env []string) []string {
	out := make([]string, 0, len(env)+1)
	prefix := llamaCppCompatEnv + "="
	for _, entry := range env {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return append(out, llamaCppCompatEnv+"=0")
}

func llamaQuantizeEnv(env []string, enableCompat bool) []string {
	if !enableCompat {
		return disableLlamaCppCompat(env)
	}

	out := make([]string, 0, len(env))
	prefix := llamaCppCompatEnv + "="
	for _, entry := range env {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return out
}

type tensorSection struct {
	*io.SectionReader
}

func (r tensorSection) WriteTo(w io.Writer) (int64, error) {
	return io.Copy(w, r.SectionReader)
}

func restoreEmbeddedCompatibilityTensors(in, out *os.File, orig *fsggml.GGML, newFileType fsggml.FileType) error {
	if _, err := out.Seek(0, io.SeekStart); err != nil {
		return err
	}

	rewritten, err := fsggml.Decode(out, -1)
	if err != nil {
		return err
	}

	kv := maps.Clone(orig.KV())
	kv["general.file_type"] = newFileType

	present := map[string]struct{}{}
	tensors := make([]*fsggml.Tensor, 0, len(rewritten.Tensors().Items()))
	for _, tensor := range rewritten.Tensors().Items() {
		if isEmbeddedCompatibilityTensor(tensor.Name) {
			continue
		}
		present[tensor.Name] = struct{}{}
		tensors = append(tensors, tensorFromFile(out, rewritten.Tensors().Offset+tensor.Offset, tensor))
	}

	var restored int
	for _, tensor := range orig.Tensors().Items() {
		if !isEmbeddedCompatibilityTensor(tensor.Name) {
			continue
		}
		if _, ok := present[tensor.Name]; ok {
			continue
		}
		tensors = append(tensors, tensorFromFile(in, orig.Tensors().Offset+tensor.Offset, tensor))
		restored++
	}
	if restored == 0 {
		return nil
	}

	temp, err := os.CreateTemp(filepath.Dir(out.Name()), "compat-tensors")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	if err := fsggml.WriteGGUF(temp, kv, tensors); err != nil {
		return err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := out.Truncate(0); err != nil {
		return err
	}
	if _, err := out.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := io.Copy(out, temp); err != nil {
		return err
	}

	slog.Info("restored embedded compatibility tensors after llama-quantize", "count", restored)
	return nil
}

func tensorFromFile(file *os.File, offset uint64, tensor *fsggml.Tensor) *fsggml.Tensor {
	return &fsggml.Tensor{
		Name:  tensor.Name,
		Kind:  tensor.Kind,
		Shape: append([]uint64(nil), tensor.Shape...),
		WriterTo: tensorSection{
			SectionReader: io.NewSectionReader(file, int64(offset), int64(tensor.Size())),
		},
	}
}

func llamaQuantizeArgs(arch string, newFileType fsggml.FileType, input, output, typeName string) []string {
	args := []string{"--allow-requantize"}
	if typeName == "COPY" {
		return append(args, input, output, typeName)
	}
	// Qwen3.5 MTP uses this projection to combine hidden and embedding states
	// for the draft layer. Keep it at least Q8 when quantizing lower than Q8,
	// while preserving unquantized outputs such as F16/BF16.
	if arch == "qwen35" || arch == "qwen35moe" {
		switch newFileType {
		case fsggml.FileTypeQ4_K_S, fsggml.FileTypeQ4_K_M:
			args = append(args, "--tensor-type", `^blk\.[0-9]+\.nextn\.eh_proj\.weight$=q8_0`)
		}
	}
	// gemma3n's per_layer_token_embd is read on every layer for every token
	// (not just once at input like token_embd), so it's far more quality-sensitive
	// than a normal token embedding. Keep it at F16 on K-quants via an anchored
	// regex so we don't also bump token_embd (which --token-embedding-type would).
	if arch == "gemma3n" {
		switch newFileType {
		case fsggml.FileTypeQ4_K_S, fsggml.FileTypeQ4_K_M:
			args = append(args, "--tensor-type", `^per_layer_token_embd\.weight$=f16`)
		}
	}
	// deepseek2 MLA tensors are small, critical matrices in DeepSeek-V2-style
	// multi-head latent attention. Keep the same higher-precision policy used
	// by published library/glm-4.7-flash K-quant models.
	if arch == "deepseek2" {
		switch newFileType {
		case fsggml.FileTypeQ4_K_S, fsggml.FileTypeQ4_K_M:
			args = append(args,
				"--tensor-type", `attn_k_b\.weight$=q8_0`,
				"--tensor-type", `attn_q_a\.weight$=q8_0`,
				"--tensor-type", `attn_q_b\.weight$=q8_0`,
				"--tensor-type", `attn_v_b\.weight$=q8_0`,
				"--tensor-type", `attn_kv_a_mqa\.weight$=q8_0`,
			)
		}
	}
	// GLM-OCR is a small multimodal OCR model; keeping the input/output
	// embeddings high precision avoids degenerate text output on K-quants.
	// Legacy Ollama GGUFs use "glmocr"; split text GGUFs use llama.cpp's
	// native "glm4" architecture.
	if arch == "glmocr" || arch == "glm4" {
		switch newFileType {
		case fsggml.FileTypeQ4_K_S, fsggml.FileTypeQ4_K_M:
			args = append(args,
				"--tensor-type", `^token_embd\.weight$=f16`,
				"--tensor-type", `^output\.weight$=f16`,
			)
		}
	}
	return append(args, input, output, typeName)
}

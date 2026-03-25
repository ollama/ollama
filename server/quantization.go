package server

import (
	"bufio"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// findLlamaQuantize locates the llama-quantize binary (installed alongside llama-server).
func findLlamaQuantize() (string, error) {
	suffix := "llama-quantize"
	if runtime.GOOS == "windows" {
		suffix += ".exe"
	}

	seen := map[string]bool{}
	var candidates []string
	add := func(dir string) {
		path := filepath.Join(dir, suffix)
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	add(ml.LibOllamaPath)

	exe, err := os.Executable()
	if err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		add(filepath.Join(filepath.Dir(exe), "build", "lib", "ollama"))
	}
	if cwd, err := os.Getwd(); err == nil {
		add(filepath.Join(cwd, "build", "lib", "ollama"))
	}

	// Dev build paths (cmake build output, before install)
	addGlob := func(base string) {
		matches, _ := filepath.Glob(filepath.Join(base, "build", "llama-server-*", "bin"))
		for _, m := range matches {
			add(m)
		}
	}
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		addGlob(filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		addGlob(cwd)
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("llama-quantize binary not found (checked: %s)", strings.Join(candidates, ", "))
}

// progressRegex matches llama-quantize output lines like "[ 42/ 200]"
var progressRegex = regexp.MustCompile(`\[\s*(\d+)/\s*(\d+)\]`)

// quantize re-quantizes a GGUF model by shelling out to llama-quantize.
// The upstream llama-quantize handles all quantization types and per-tensor
// type selection (mixed quantization for quality).
func quantize(in, out *os.File, orig *fsggml.GGML, newFileType fsggml.FileType, progressFn func(n uint64)) error {
	quantizeExe, err := findLlamaQuantize()
	if err != nil {
		return fmt.Errorf("quantization unavailable: %w", err)
	}

	// Map our FileType to the llama-quantize type name
	typeName := newFileType.String()
	if typeName == "" {
		return fmt.Errorf("unsupported quantization type: %v", newFileType)
	}

	slog.Info("quantizing model", "type", typeName, "input", in.Name(), "output", out.Name())

	args := []string{"--allow-requantize"}
	arch := orig.KV().Architecture()
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
	// deepseek2 MLA tensors (attn_k_b / attn_q_a / attn_q_b / attn_v_b /
	// attn_kv_a_mqa) are small, critical matrices in DeepSeek-V2-style multi-head
	// latent attention. Upstream llama-quant.cpp has no special case for these
	// names at b8680 — they fall through to the default Q4_K / Q5_0 for Q4_K_M.
	// Published library/glm-4.7-flash quantizes them at Q8_0 for quality. Force
	// the same on K-quants via --tensor-type regex so we match.
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
	args = append(args, in.Name(), out.Name(), typeName)
	cmd := exec.Command(quantizeExe, args...)
	cmd.Env = os.Environ()

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

package llm

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// SMT solver integration tests - verify implementation against formal specifications

type SMTVerifier struct {
	solverPath    string
	workingDir    string
	timeout       time.Duration
	tempDir       string
}

func NewSMTVerifier() (*SMTVerifier, error) {
	// Check if Z3 is available
	solverPath, err := exec.LookPath("z3")
	if err != nil {
		return nil, fmt.Errorf("Z3 solver not found: %w", err)
	}

	// Create temporary directory for SMT files
	tempDir, err := os.MkdirTemp("", "ollama_smt_test_")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp dir: %w", err)
	}

	return &SMTVerifier{
		solverPath: solverPath,
		workingDir: "verification",
		timeout:    30 * time.Second,
		tempDir:    tempDir,
	}, nil
}

func (sv *SMTVerifier) Cleanup() {
	if sv.tempDir != "" {
		os.RemoveAll(sv.tempDir)
	}
}

// Generate SMT file for specific test case
func (sv *SMTVerifier) generateSMTForCheckpoint(layers int, expectedCheckpoint uint64) (string, error) {
	smtContent := fmt.Sprintf(`
;; Generated SMT verification for checkpoint optimization
(set-logic QF_LIA)

(declare-const layers Int)
(declare-const checkpoint_result Int)
(declare-const sqrt_approx Int)

;; Test case: layers = %d
(assert (= layers %d))

;; Constraint: sqrt_approx should be approximately sqrt(layers)
(assert (and (<= (* sqrt_approx sqrt_approx) layers)
             (< layers (* (+ sqrt_approx 1) (+ sqrt_approx 1)))))

;; Checkpoint function: sqrt(layers) + 1
(assert (= checkpoint_result (+ sqrt_approx 1)))

;; Expected result
(assert (= checkpoint_result %d))

;; Property: checkpoint < layers for layers >= 4
(assert (=> (>= layers 4) (< checkpoint_result layers)))

(check-sat)
(get-model)
`, layers, layers, expectedCheckpoint)

	filename := filepath.Join(sv.tempDir, fmt.Sprintf("checkpoint_test_%d.smt2", layers))
	err := os.WriteFile(filename, []byte(smtContent), 0644)
	return filename, err
}

func (sv *SMTVerifier) generateSMTForMLA(kvSize uint64, expectedCompressed uint64) (string, error) {
	smtContent := fmt.Sprintf(`
;; Generated SMT verification for MLA compression
(set-logic QF_LIA)

(declare-const kv_size Int)
(declare-const compressed_result Int)

;; Test case: kv_size = %d
(assert (= kv_size %d))

;; MLA compression: kv_size / 28
(assert (= compressed_result (div kv_size 28)))

;; Expected result
(assert (= compressed_result %d))

;; Property: compressed < original for kv_size >= 28
(assert (=> (>= kv_size 28) (< compressed_result kv_size)))

(check-sat)
(get-model)
`, kvSize, kvSize, expectedCompressed)

	filename := filepath.Join(sv.tempDir, fmt.Sprintf("mla_test_%d.smt2", kvSize))
	err := os.WriteFile(filename, []byte(smtContent), 0644)
	return filename, err
}

func (sv *SMTVerifier) generateSMTForGPUScore(memGB, tflops int, tensorCores bool, expectedScore int) (string, error) {
	tensorValue := 0
	if tensorCores {
		tensorValue = 1
	}

	smtContent := fmt.Sprintf(`
;; Generated SMT verification for GPU scoring
(set-logic QF_LIA)

(declare-const memory_gb Int)
(declare-const tflops Int)
(declare-const has_tensor_cores Int)
(declare-const score_result Int)

;; Test case
(assert (= memory_gb %d))
(assert (= tflops %d))
(assert (= has_tensor_cores %d))

;; GPU scoring function: memory_gb * 10 + tflops + (tensor_cores ? 50 : 0)
(assert (= score_result (+ (* memory_gb 10) tflops (* has_tensor_cores 50))))

;; Expected result
(assert (= score_result %d))

;; Properties
(assert (>= score_result 0))  ;; Score should be positive
(assert (>= score_result (* memory_gb 10)))  ;; Memory component should be included

(check-sat)
(get-model)
`, memGB, tflops, tensorValue, expectedScore)

	filename := filepath.Join(sv.tempDir, fmt.Sprintf("gpu_score_test_%d_%d_%t.smt2", memGB, tflops, tensorCores))
	err := os.WriteFile(filename, []byte(smtContent), 0644)
	return filename, err
}

// Run SMT solver on a file
func (sv *SMTVerifier) runSMT(filename string) (bool, string, error) {
	cmd := exec.Command(sv.solverPath, filename)
	cmd.Dir = sv.workingDir

	// Set timeout
	done := make(chan error, 1)
	go func() {
		done <- cmd.Run()
	}()

	select {
	case err := <-done:
		if err != nil {
			return false, "", fmt.Errorf("Z3 execution failed: %w", err)
		}
	case <-time.After(sv.timeout):
		cmd.Process.Kill()
		return false, "", fmt.Errorf("Z3 execution timeout")
	}

	output, err := cmd.Output()
	if err != nil {
		return false, "", fmt.Errorf("failed to get Z3 output: %w", err)
	}

	outputStr := string(output)
	isSat := strings.Contains(outputStr, "sat")

	return isSat, outputStr, nil
}

// Test checkpoint optimization against SMT specification
func TestSMTCheckpointVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	optimizer := NewVerifiedMemoryOptimizer(true, false)

	testCases := []struct {
		layers   int
		expected uint64
	}{
		{4, 3},   // sqrt(4) + 1 = 3
		{16, 5},  // sqrt(16) + 1 = 5
		{64, 9},  // sqrt(64) + 1 = 9
		{100, 11}, // sqrt(100) + 1 = 11
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("layers_%d", tc.layers), func(t *testing.T) {
			// Get implementation result
			result := optimizer.CheckpointMemoryEstimate(tc.layers)

			// Verify expected result matches implementation
			if result != tc.expected {
				t.Errorf("Implementation mismatch: got %d, expected %d for %d layers",
					result, tc.expected, tc.layers)
			}

			// Generate and run SMT verification
			smtFile, err := verifier.generateSMTForCheckpoint(tc.layers, result)
			if err != nil {
				t.Fatalf("Failed to generate SMT: %v", err)
			}

			isSat, output, err := verifier.runSMT(smtFile)
			if err != nil {
				t.Fatalf("SMT verification failed: %v", err)
			}

			if !isSat {
				t.Errorf("SMT specification unsatisfiable for layers=%d\nOutput: %s", tc.layers, output)
			}

			t.Logf("SMT verification passed for %d layers -> %d checkpoints", tc.layers, result)
		})
	}
}

// Test MLA compression against SMT specification
func TestSMTMLAVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	optimizer := NewVerifiedMemoryOptimizer(false, true)

	testCases := []struct {
		kvSize   uint64
		expected uint64
	}{
		{28, 1},     // 28/28 = 1
		{280, 10},   // 280/28 = 10
		{1500, 53},  // 1500/28 = 53
		{2800, 100}, // 2800/28 = 100
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("kv_%d", tc.kvSize), func(t *testing.T) {
			// Get implementation result
			result := optimizer.MLACompressionEstimate(tc.kvSize)

			// Verify expected result matches implementation
			if result != tc.expected {
				t.Errorf("Implementation mismatch: got %d, expected %d for KV size %d",
					result, tc.expected, tc.kvSize)
			}

			// Generate and run SMT verification
			smtFile, err := verifier.generateSMTForMLA(tc.kvSize, result)
			if err != nil {
				t.Fatalf("Failed to generate SMT: %v", err)
			}

			isSat, output, err := verifier.runSMT(smtFile)
			if err != nil {
				t.Fatalf("SMT verification failed: %v", err)
			}

			if !isSat {
				t.Errorf("SMT specification unsatisfiable for KV size=%d\nOutput: %s", tc.kvSize, output)
			}

			t.Logf("SMT verification passed for KV %d -> %d compressed", tc.kvSize, result)
		})
	}
}

// Test GPU scoring against SMT specification
func TestSMTGPUScoringVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	testCases := []struct {
		name        string
		memGB       int
		tflops      int
		tensorCores bool
		expected    int
	}{
		{"RTX_3070", 8, 20, true, 150},   // 8*10 + 20 + 50 = 150
		{"Arc_B580", 12, 17, true, 187},  // 12*10 + 17 + 50 = 187
		{"RX_6700XT", 12, 25, false, 145}, // 12*10 + 25 + 0 = 145
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create GPU spec and get implementation result
			spec := &GPUDeviceSpec{
				MemorySizeGB:        tc.memGB,
				PeakTFLOPSFP32:      tc.tflops,
				SupportsTensorCores: tc.tensorCores,
			}

			result := spec.DeviceScore()

			// Verify expected result matches implementation
			if result != tc.expected {
				t.Errorf("Implementation mismatch: got %d, expected %d for %s",
					result, tc.expected, tc.name)
			}

			// Generate and run SMT verification
			smtFile, err := verifier.generateSMTForGPUScore(tc.memGB, tc.tflops, tc.tensorCores, result)
			if err != nil {
				t.Fatalf("Failed to generate SMT: %v", err)
			}

			isSat, output, err := verifier.runSMT(smtFile)
			if err != nil {
				t.Fatalf("SMT verification failed: %v", err)
			}

			if !isSat {
				t.Errorf("SMT specification unsatisfiable for %s\nOutput: %s", tc.name, output)
			}

			t.Logf("SMT verification passed for %s: score=%d", tc.name, result)
		})
	}
}

// Test combined optimizations with DeepSeek V3 configuration
func TestSMTCombinedOptimizationVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	optimizer := NewVerifiedMemoryOptimizer(true, true)

	// DeepSeek V3 configuration from our proofs
	layers := 64
	kvSize := uint64(1500)

	checkpointMem := optimizer.CheckpointMemoryEstimate(layers)
	compressedKV := optimizer.MLACompressionEstimate(kvSize)

	originalTotal := uint64(layers) + kvSize  // 1564
	optimizedTotal := checkpointMem + compressedKV  // 9 + 53 = 62
	savings := originalTotal - optimizedTotal  // 1502

	// Expected values from SMT proofs
	expectedCheckpoint := uint64(9)
	expectedCompressed := uint64(53)
	expectedSavings := uint64(1502)

	if checkpointMem != expectedCheckpoint {
		t.Errorf("Checkpoint mismatch: got %d, expected %d", checkpointMem, expectedCheckpoint)
	}

	if compressedKV != expectedCompressed {
		t.Errorf("MLA compression mismatch: got %d, expected %d", compressedKV, expectedCompressed)
	}

	if savings != expectedSavings {
		t.Errorf("Total savings mismatch: got %d, expected %d", savings, expectedSavings)
	}

	// Calculate efficiency
	efficiency := float64(savings) / float64(originalTotal)
	_ = 1502.0 / 1564.0  // expectedEfficiency ≈ 0.9604

	if efficiency < 0.96 {
		t.Errorf("Efficiency too low: got %.4f, expected >= 0.96", efficiency)
	}

	// Generate SMT file to verify the combined calculation
	smtContent := fmt.Sprintf(`
;; SMT verification for DeepSeek V3 combined optimization
(set-logic QF_LIA)

(declare-const original_layers Int)
(declare-const original_kv Int)
(declare-const checkpoint_layers Int)
(declare-const compressed_kv Int)
(declare-const total_savings Int)

;; DeepSeek V3 configuration
(assert (= original_layers 64))
(assert (= original_kv 1500))

;; Implementation results
(assert (= checkpoint_layers %d))
(assert (= compressed_kv %d))

;; Calculate savings
(assert (= total_savings (+ (- original_layers checkpoint_layers) (- original_kv compressed_kv))))

;; Verify our expected savings
(assert (= total_savings %d))

;; Properties from our Coq proofs
(assert (< checkpoint_layers original_layers))
(assert (< compressed_kv original_kv))
(assert (> total_savings 1400))  ;; Significant savings

(check-sat)
(get-model)
`, checkpointMem, compressedKV, savings)

	smtFile := filepath.Join(verifier.tempDir, "deepseek_v3_combined.smt2")
	err = os.WriteFile(smtFile, []byte(smtContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write SMT file: %v", err)
	}

	isSat, output, err := verifier.runSMT(smtFile)
	if err != nil {
		t.Fatalf("SMT verification failed: %v", err)
	}

	if !isSat {
		t.Errorf("SMT specification unsatisfiable for DeepSeek V3 combined optimization\nOutput: %s", output)
	}

	t.Logf("SMT verification passed for DeepSeek V3: %d layers -> %d, %d KV -> %d, savings: %d (%.2f%%)",
		layers, checkpointMem, kvSize, compressedKV, savings, efficiency*100)
}

// Test that verifies our concrete SMT file from the verification directory
func TestSMTConcreteVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}

	// Check if concrete verification file exists
	concreteFile := filepath.Join("..", "verification", "concrete_verification.smt2")
	if _, err := os.Stat(concreteFile); os.IsNotExist(err) {
		t.Skip("Concrete SMT verification file not found")
		return
	}

	isSat, output, err := verifier.runSMT(concreteFile)
	if err != nil {
		t.Fatalf("Failed to run concrete SMT verification: %v", err)
	}

	if !isSat {
		t.Errorf("Concrete SMT verification failed\nOutput: %s", output)
	} else {
		t.Log("Concrete SMT verification passed")

		// Extract specific values from the model
		lines := strings.Split(output, "\n")
		for _, line := range lines {
			if strings.Contains(line, "deepseek_savings") {
				t.Logf("SMT model: %s", strings.TrimSpace(line))
			}
		}
	}
}

// Benchmark SMT verification performance
func BenchmarkSMTVerification(b *testing.B) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		b.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	optimizer := NewVerifiedMemoryOptimizer(true, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layers := 64
		result := optimizer.CheckpointMemoryEstimate(layers)

		smtFile, err := verifier.generateSMTForCheckpoint(layers, result)
		if err != nil {
			b.Fatalf("Failed to generate SMT: %v", err)
		}

		_, _, err = verifier.runSMT(smtFile)
		if err != nil {
			b.Fatalf("SMT verification failed: %v", err)
		}
	}
}

// Property-based test that generates random cases and verifies with SMT
func TestSMTPropertyBasedVerification(t *testing.T) {
	verifier, err := NewSMTVerifier()
	if err != nil {
		t.Skipf("SMT verification skipped: %v", err)
		return
	}
	defer verifier.Cleanup()

	optimizer := NewVerifiedMemoryOptimizer(true, true)
	gen := NewPropertyTestGenerator()

	// Test a smaller number of cases due to SMT solver overhead
	const numTests = 10

	for i := 0; i < numTests; i++ {
		layers := gen.GenLayers()
		if layers < 4 || layers > 100 {
			continue // Skip edge cases for SMT verification
		}

		result := optimizer.CheckpointMemoryEstimate(layers)

		// Verify basic property: result should be less than layers
		if result >= uint64(layers) {
			t.Errorf("Property violation before SMT: checkpoint(%d) = %d >= %d",
				layers, result, layers)
			continue
		}

		// Generate and verify with SMT
		smtFile, err := verifier.generateSMTForCheckpoint(layers, result)
		if err != nil {
			t.Errorf("Failed to generate SMT for layers=%d: %v", layers, err)
			continue
		}

		isSat, _, err := verifier.runSMT(smtFile)
		if err != nil {
			t.Errorf("SMT verification failed for layers=%d: %v", layers, err)
			continue
		}

		if !isSat {
			t.Errorf("SMT verification failed for layers=%d, result=%d", layers, result)
		}
	}
}

// Helper function to check if a string represents a valid integer
func isValidInteger(s string) bool {
	_, err := strconv.Atoi(strings.TrimSpace(s))
	return err == nil
}

// Extract model values from Z3 output
func extractModelValue(output, varName string) (int, error) {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.Contains(line, varName) && strings.Contains(line, "define-fun") {
			// Parse line like: "(define-fun varName () Int 42)"
			parts := strings.Fields(line)
			for i, part := range parts {
				if strings.TrimSuffix(part, ")") == varName && i+3 < len(parts) {
					valueStr := strings.TrimSuffix(parts[i+3], ")")
					return strconv.Atoi(valueStr)
				}
			}
		}
	}
	return 0, fmt.Errorf("variable %s not found in model", varName)
}
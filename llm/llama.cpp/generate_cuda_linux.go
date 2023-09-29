//go:build exclude

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func main() {
	generateGGML(findOpenCL(), findCUDA(), findROCM())
	generateGGUF(findOpenCL(), findCUDA(), findROCM())
}

func generateGGUF(hasOCL, hasCUDA, hasROCM bool) {
	cmd := []string{
		"cmake",
		"-S",
		"gguf",
		"-B",
		"gguf/build/cuda",
		"-DLLAMA_K_QUANTS=on",
		"-DLLAMA_ACCELERATE=on", // Should be on by default for GGUF, but here for certainty.
	}
	if hasCUDA {
		log.Printf("CUDA found!")
		cmd = append(cmd, "-DLLAMA_CUBLAS=on")
	} else if hasROCM {
		log.Printf("ROCM found!!")
		cmd = append(cmd, "-DLLAMA_HIPBLAS=on")
		cmd = append(cmd, fmt.Sprintf("-DCMAKE_C_COMPILER=%s", filepath.Join(getROCMHome(), "llvm/bin/clang")))
		cmd = append(cmd, fmt.Sprintf("-DCMAKE_CXX_COMPILER=%s", filepath.Join(getROCMHome(), "llvm/bin/clang++")))
	} else if hasOCL {
		log.Printf("OpenCL found!!")
		cmd = append(cmd, "-DLLAMA_CLBLAST=on")
	}
	cmake := exec.Command(cmd[0], cmd[1:]...)
	cmake.Stdout = os.Stdout
	cmake.Stderr = os.Stderr
	if err := cmake.Run(); err != nil {
		log.Fatalf("CMake generation failed: %v", err)
	}
}

// No ROCm support in GGML.
func generateGGML(hasOCL, hasCUDA, hasROCM bool) {
	cmd := []string{
		"cmake",
		"-S",
		"ggml",
		"-B",
		"ggml/build/cuda",
		"-DLLAMA_K_QUANTS=on",
		"-DLLAMA_ACCELERATE=on",
	}
	if hasCUDA {
		log.Printf("CUDA found!")
		cmd = append(cmd, "-DLLAMA_CUBLAS=on")
	} else if hasOCL {
		log.Printf("OpenCL found!!")
		cmd = append(cmd, "-DLLAMA_CLBLAST=on")
	}
	cmake := exec.Command(cmd[0], cmd[1:]...)
	cmake.Stdout = os.Stdout
	cmake.Stderr = os.Stderr
	if err := cmake.Run(); err != nil {
		log.Fatalf("CMake generation failed: %v", err)
	}
}

func findOpenCL() bool {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(10)*time.Second)
	defer cancel()
	_, err := exec.CommandContext(ctx, "clinfo").CombinedOutput()
	return err == nil
}

func findCUDA() bool {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(10)*time.Second)
	defer cancel()
	_, err := exec.CommandContext(ctx, "nvidia-smi").CombinedOutput()
	return err == nil
}

func getROCMHome() string {
	rocmHome := os.Getenv("ROCM_HOME")
	if rocmHome != "" {
		return rocmHome
	}
	return "/opt/rocm"
}

func findROCM() bool {
	if _, err := os.Stat(getROCMHome()); err != nil && !os.IsNotExist(err) {
		return false
	}
	return true
}

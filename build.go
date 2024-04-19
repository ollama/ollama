//go:build ignore

package main

import (
	"cmp"
	"errors"
	"flag"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// Flags
var (
	flagRegenerateDestroy = flag.Bool("d", false, "force regenerate the dependencies (destructive)")
	flagRegenerateGently  = flag.Bool("g", false, "regenerate the dependencies (non-destructive)")
	flagSkipBuild         = flag.Bool("s", false, "generate dependencies only (e.g. skip 'go build .')")

	// Flags to set GOARCH explicitly for cross-platform builds,
	// e.g., in CI to target a different platform than the build matrix
	// default. These allows us to run generate without a separate build
	// step for building the script binary for the host ARCH and then
	// runing the generate script for the target ARCH. Instead, we can
	// just run `go run build.go -target=$GOARCH` to generate the
	// deps.
	flagGOARCH = flag.String("target", "", "sets GOARCH to use when generating dependencies and building")
)

func buildEnv() []string {
	return append(os.Environ(), "GOARCH="+cmp.Or(
		*flagGOARCH,
		os.Getenv("OLLAMA_BUILD_TARGET_ARCH"),
		runtime.GOARCH,
	))
}

func main() {
	log.SetFlags(0)
	flag.Usage = func() {
		log.Printf("Usage: go run build.go [flags]")
		log.Println()
		log.Println("Flags:")
		flag.PrintDefaults()
		log.Println()
		log.Println("This script builds the Ollama server binary and generates the llama.cpp")
		log.Println("bindings for the current platform. It assumes that the current working")
		log.Println("directory is the root directory of the Ollama project.")
		log.Println()
		log.Println("If the -d flag is provided, the script will force regeneration of the")
		log.Println("dependencies; removing the 'llm/build' directory before starting.")
		log.Println()
		log.Println("If the -g flag is provided, the script will regenerate the dependencies")
		log.Println("without removing the 'llm/build' directory.")
		log.Println()
		log.Println("If the -s flag is provided, the script will skip building the Ollama binary")
		log.Println()
		log.Println("If the -target flag is provided, the script will set GOARCH to the value")
		log.Println("of the flag. This is useful for cross-platform builds.")
		log.Println()
		log.Println("The script will check for the required dependencies (cmake, gcc) and")
		log.Println("print their version.")
		log.Println()
		log.Println("The script will also check if it is being run from the root directory of")
		log.Println("the Ollama project.")
		log.Println()
		os.Exit(1)
	}
	flag.Parse()

	log.Printf("=== Building Ollama ===")
	defer func() {
		log.Printf("=== Done building Ollama ===")
		if !*flagSkipBuild {
			log.Println()
			log.Println("To run the Ollama server, use:")
			log.Println()
			log.Println("    ./ollama serve")
			log.Println()
		}
	}()

	if flag.NArg() > 0 {
		flag.Usage()
	}

	if !inRootDir() {
		log.Fatalf("Please run this script from the root directory of the Ollama project.")
	}

	if err := checkDependencies(); err != nil {
		log.Fatalf("Failed dependency check: %v", err)
	}
	if err := buildLlammaCPP(); err != nil {
		log.Fatalf("Failed to build llama.cpp: %v", err)
	}
	if err := goBuildOllama(); err != nil {
		log.Fatalf("Failed to build ollama Go binary: %v", err)
	}
}

// checkDependencies does a quick check to see if the required dependencies are
// installed on the system and functioning enough to print their version.
//
// TODO(bmizerany): Check the actual version of the dependencies? Seems a
// little daunting given diff versions might print diff things. This should
// be good enough for now.
func checkDependencies() error {
	var err error
	check := func(name string, args ...string) {
		log.Printf("=== Checking for %s ===", name)
		defer log.Printf("=== Done checking for %s ===\n\n", name)
		cmd := exec.Command(name, args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err = errors.Join(err, cmd.Run())
	}

	check("cmake", "--version")
	check("gcc", "--version")
	return err
}

func goBuildOllama() error {
	log.Println("=== Building Ollama binary ===")
	defer log.Printf("=== Done building Ollama binary ===\n\n")
	if *flagSkipBuild {
		log.Println("Skipping 'go build -o ollama .'")
		return nil
	}
	cmd := exec.Command("go", "build", "-o", "ollama", ".")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildEnv()
	return cmd.Run()
}

// buildLlammaCPP generates the llama.cpp bindings for the current platform.
//
// It assumes that the current working directory is the root directory of the
// Ollama project.
func buildLlammaCPP() error {
	log.Println("=== Generating dependencies ===")
	defer log.Printf("=== Done generating dependencies ===\n\n")
	if *flagRegenerateDestroy {
		if err := os.RemoveAll(filepath.Join("llm", "build")); err != nil {
			return err
		}
	}
	if isDirectory(filepath.Join("llm", "build")) && !*flagRegenerateGently {
		log.Println("llm/build already exists; skipping.  Use -d or -g to re-generate.")
		return nil
	}

	scriptDir, err := filepath.Abs(filepath.Join("llm", "generate"))
	if err != nil {
		return err
	}

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		script := filepath.Join(scriptDir, "gen_windows.ps1")
		cmd = exec.Command("powershell", "-ExecutionPolicy", "Bypass", "-File", script)
	case "linux":
		script := filepath.Join(scriptDir, "gen_linux.sh")
		cmd = exec.Command("bash", script)
	case "darwin":
		script := filepath.Join(scriptDir, "gen_darwin.sh")
		cmd = exec.Command("bash", script)
	default:
		log.Fatalf("Unsupported OS: %s", runtime.GOOS)
	}
	cmd.Dir = filepath.Join("llm", "generate")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = buildEnv()

	log.Printf("Running GOOS=%s GOARCH=%s %s", runtime.GOOS, runtime.GOARCH, cmd.Args)

	return cmd.Run()
}

func isDirectory(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// inRootDir returns true if the current working directory is the root
// directory of the Ollama project. It looks for a file named "go.mod".
func inRootDir() bool {
	_, err := os.Stat("go.mod")
	return err == nil
}

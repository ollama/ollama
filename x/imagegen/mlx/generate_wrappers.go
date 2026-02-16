//go:build ignore

// This tool generates MLX-C dynamic loading wrappers.
// Usage: go run generate_wrappers.go <mlx-c-include-dir> <output-header> [output-impl]
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Function struct {
	Name          string
	ReturnType    string
	Params        string
	ParamNames    []string
	NeedsARM64Guard bool
}

func findHeaders(directory string) ([]string, error) {
	var headers []string
	err := filepath.WalkDir(directory, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".h") {
			headers = append(headers, path)
		}
		return nil
	})
	return headers, err
}

func cleanContent(content string) string {
	// Remove single-line comments
	re := regexp.MustCompile(`//.*?\n`)
	content = re.ReplaceAllString(content, "\n")

	// Remove multi-line comments
	re = regexp.MustCompile(`/\*.*?\*/`)
	content = re.ReplaceAllString(content, "")

	// Remove preprocessor directives (lines starting with #) - use multiline mode
	re = regexp.MustCompile(`(?m)^\s*#.*?$`)
	content = re.ReplaceAllString(content, "")

	// Remove extern "C" { and } blocks more conservatively
	// Only remove the extern "C" { line, not the content inside
	re = regexp.MustCompile(`extern\s+"C"\s*\{\s*?\n`)
	content = re.ReplaceAllString(content, "\n")
	// Remove standalone closing braces that are not part of function declarations
	re = regexp.MustCompile(`\n\s*\}\s*\n`)
	content = re.ReplaceAllString(content, "\n")

	// Collapse whitespace and newlines
	re = regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	return content
}

func extractParamNames(params string) []string {
	if params == "" || strings.TrimSpace(params) == "void" {
		return []string{}
	}

	var names []string

	// Split by comma, but respect parentheses (for function pointers)
	parts := splitParams(params)

	// Remove array brackets
	arrayBrackets := regexp.MustCompile(`\[.*?\]`)

	// Function pointer pattern
	funcPtrPattern := regexp.MustCompile(`\(\s*\*\s*(\w+)\s*\)`)

	// Type keywords to skip
	typeKeywords := map[string]bool{
		"const":     true,
		"struct":    true,
		"unsigned":  true,
		"signed":    true,
		"long":      true,
		"short":     true,
		"int":       true,
		"char":      true,
		"float":     true,
		"double":    true,
		"void":      true,
		"size_t":    true,
		"uint8_t":   true,
		"uint16_t":  true,
		"uint32_t":  true,
		"uint64_t":  true,
		"int8_t":    true,
		"int16_t":   true,
		"int32_t":   true,
		"int64_t":   true,
		"intptr_t":  true,
		"uintptr_t": true,
	}

	for _, part := range parts {
		if part == "" {
			continue
		}

		// Remove array brackets
		part = arrayBrackets.ReplaceAllString(part, "")

		// For function pointers like "void (*callback)(int)"
		if matches := funcPtrPattern.FindStringSubmatch(part); len(matches) > 1 {
			names = append(names, matches[1])
			continue
		}

		// Regular parameter: last identifier
		tokens := regexp.MustCompile(`\w+`).FindAllString(part, -1)
		if len(tokens) > 0 {
			// The last token is usually the parameter name
			// Skip type keywords
			for i := len(tokens) - 1; i >= 0; i-- {
				if !typeKeywords[tokens[i]] {
					names = append(names, tokens[i])
					break
				}
			}
		}
	}

	return names
}

func splitParams(params string) []string {
	var parts []string
	var current bytes.Buffer
	depth := 0

	for _, char := range params + "," {
		switch char {
		case '(':
			depth++
			current.WriteRune(char)
		case ')':
			depth--
			current.WriteRune(char)
		case ',':
			if depth == 0 {
				parts = append(parts, strings.TrimSpace(current.String()))
				current.Reset()
			} else {
				current.WriteRune(char)
			}
		default:
			current.WriteRune(char)
		}
	}

	return parts
}

func parseFunctions(content string) []Function {
	var functions []Function

	// Match function declarations: return_type function_name(params);
	// Matches both mlx_* and _mlx_* functions
	pattern := regexp.MustCompile(`\b((?:const\s+)?(?:struct\s+)?[\w\s]+?[\*\s]*)\s+(_?mlx_\w+)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*;`)

	matches := pattern.FindAllStringSubmatch(content, -1)
	for _, match := range matches {
		returnType := strings.TrimSpace(match[1])
		funcName := strings.TrimSpace(match[2])
		params := strings.TrimSpace(match[3])

		// Skip if this looks like a variable declaration
		if params == "" || strings.Contains(params, "{") {
			continue
		}

		// Clean up return type
		returnType = strings.Join(strings.Fields(returnType), " ")

		// Extract parameter names
		paramNames := extractParamNames(params)

		// Check if ARM64 guard is needed
		needsGuard := needsARM64Guard(funcName, returnType, params)

		functions = append(functions, Function{
			Name:           funcName,
			ReturnType:     returnType,
			Params:         params,
			ParamNames:     paramNames,
			NeedsARM64Guard: needsGuard,
		})
	}

	return functions
}

func needsARM64Guard(name, retType, params string) bool {
	return strings.Contains(name, "float16") ||
		strings.Contains(name, "bfloat16") ||
		strings.Contains(retType, "float16_t") ||
		strings.Contains(retType, "bfloat16_t") ||
		strings.Contains(params, "float16_t") ||
		strings.Contains(params, "bfloat16_t")
}

func generateWrapperFiles(functions []Function, headerPath, implPath string) error {
	// Generate header file
	var headerBuf bytes.Buffer

	headerBuf.WriteString("// AUTO-GENERATED by generate_wrappers.go - DO NOT EDIT\n")
	headerBuf.WriteString("// This file provides wrapper declarations for MLX-C functions that use dlopen/dlsym\n")
	headerBuf.WriteString("//\n")
	headerBuf.WriteString("// Strategy: Include MLX-C headers for type definitions, then provide wrapper\n")
	headerBuf.WriteString("// functions that shadow the originals, allowing Go code to call them directly (e.g., C.mlx_add).\n")
	headerBuf.WriteString("// Function pointers are defined in mlx.c (single compilation unit).\n\n")
	headerBuf.WriteString("#ifndef MLX_WRAPPERS_H\n")
	headerBuf.WriteString("#define MLX_WRAPPERS_H\n\n")

	headerBuf.WriteString("// Include MLX headers for type definitions and original declarations\n")
	headerBuf.WriteString("#include \"mlx/c/mlx.h\"\n")
	headerBuf.WriteString("#include \"mlx_dynamic.h\"\n")
	headerBuf.WriteString("#include <stdio.h>\n\n")

	// Undef all MLX functions to avoid conflicts
	headerBuf.WriteString("// Undefine any existing MLX function macros\n")
	for _, fn := range functions {
		headerBuf.WriteString(fmt.Sprintf("#undef %s\n", fn.Name))
	}
	headerBuf.WriteString("\n")

	// Function pointer extern declarations
	headerBuf.WriteString("// Function pointer declarations (defined in mlx.c, loaded via dlsym)\n")
	for _, fn := range functions {
		if fn.NeedsARM64Guard {
			headerBuf.WriteString("#if defined(__aarch64__) || defined(_M_ARM64)\n")
		}
		headerBuf.WriteString(fmt.Sprintf("extern %s (*%s_ptr)(%s);\n", fn.ReturnType, fn.Name, fn.Params))
		if fn.NeedsARM64Guard {
			headerBuf.WriteString("#endif\n")
		}
	}
	headerBuf.WriteString("\n")

	// Initialization function declaration
	headerBuf.WriteString("// Initialize all function pointers via dlsym (defined in mlx.c)\n")
	headerBuf.WriteString("int mlx_load_functions(void* handle);\n\n")

	// Wrapper function declarations
	headerBuf.WriteString("// Wrapper function declarations that call through function pointers\n")
	headerBuf.WriteString("// Go code calls these directly as C.mlx_* (no #define redirection needed)\n")
	for _, fn := range functions {
		if fn.NeedsARM64Guard {
			headerBuf.WriteString("#if defined(__aarch64__) || defined(_M_ARM64)\n")
		}
		headerBuf.WriteString(fmt.Sprintf("%s %s(%s);\n", fn.ReturnType, fn.Name, fn.Params))
		if fn.NeedsARM64Guard {
			headerBuf.WriteString("#endif\n")
		}
		headerBuf.WriteString("\n")
	}

	headerBuf.WriteString("#endif // MLX_WRAPPERS_H\n")

	// Write header file
	if err := os.WriteFile(headerPath, headerBuf.Bytes(), 0644); err != nil {
		return fmt.Errorf("failed to write header file: %w", err)
	}

	// Generate implementation file
	var implBuf bytes.Buffer

	implBuf.WriteString("// AUTO-GENERATED by generate_wrappers.go - DO NOT EDIT\n")
	implBuf.WriteString("// This file contains the function pointer definitions and initialization\n")
	implBuf.WriteString("// All function pointers are in a single compilation unit to avoid duplication\n\n")

	implBuf.WriteString("#include \"mlx/c/mlx.h\"\n")
	implBuf.WriteString("#include \"mlx_dynamic.h\"\n")
	implBuf.WriteString("#include <stdio.h>\n")
	implBuf.WriteString("#include <dlfcn.h>\n\n")

	// Function pointer definitions
	implBuf.WriteString("// Function pointer definitions\n")
	for _, fn := range functions {
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#if defined(__aarch64__) || defined(_M_ARM64)\n")
		}
		implBuf.WriteString(fmt.Sprintf("%s (*%s_ptr)(%s) = NULL;\n", fn.ReturnType, fn.Name, fn.Params))
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#endif\n")
		}
	}
	implBuf.WriteString("\n")

	// Initialization function
	implBuf.WriteString("// Initialize all function pointers via dlsym\n")
	implBuf.WriteString("int mlx_load_functions(void* handle) {\n")
	implBuf.WriteString("    if (handle == NULL) {\n")
	implBuf.WriteString("        fprintf(stderr, \"MLX: Invalid library handle\\n\");\n")
	implBuf.WriteString("        return -1;\n")
	implBuf.WriteString("    }\n\n")

	for _, fn := range functions {
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#if defined(__aarch64__) || defined(_M_ARM64)\n")
		}
		implBuf.WriteString(fmt.Sprintf("    %s_ptr = dlsym(handle, \"%s\");\n", fn.Name, fn.Name))
		implBuf.WriteString(fmt.Sprintf("    if (%s_ptr == NULL) {\n", fn.Name))
		implBuf.WriteString(fmt.Sprintf("        fprintf(stderr, \"MLX: Failed to load symbol: %s\\n\");\n", fn.Name))
		implBuf.WriteString("        return -1;\n")
		implBuf.WriteString("    }\n")
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#endif\n")
		}
	}

	implBuf.WriteString("    return 0;\n")
	implBuf.WriteString("}\n\n")

	// Wrapper function implementations
	implBuf.WriteString("// Wrapper function implementations that call through function pointers\n")
	for _, fn := range functions {
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#if defined(__aarch64__) || defined(_M_ARM64)\n")
		}
		implBuf.WriteString(fmt.Sprintf("%s %s(%s) {\n", fn.ReturnType, fn.Name, fn.Params))

		// Call through function pointer
		if fn.ReturnType != "void" {
			implBuf.WriteString(fmt.Sprintf("    return %s_ptr(", fn.Name))
		} else {
			implBuf.WriteString(fmt.Sprintf("    %s_ptr(", fn.Name))
		}

		// Pass parameters
		implBuf.WriteString(strings.Join(fn.ParamNames, ", "))
		implBuf.WriteString(");\n")
		implBuf.WriteString("}\n")
		if fn.NeedsARM64Guard {
			implBuf.WriteString("#endif\n")
		}
		implBuf.WriteString("\n")
	}

	// Write implementation file
	if err := os.WriteFile(implPath, implBuf.Bytes(), 0644); err != nil {
		return fmt.Errorf("failed to write implementation file: %w", err)
	}

	return nil
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: go run generate_wrappers.go <mlx-c-include-dir> <output-header> [output-impl]\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Generate MLX-C dynamic loading wrappers.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		fmt.Fprintf(flag.CommandLine.Output(), "ERROR: Missing required arguments\n\n")
		flag.Usage()
		os.Exit(1)
	}

	headerDir := args[0]
	outputHeader := args[1]
	// Default implementation file is same name with .c extension
	outputImpl := outputHeader
	if len(args) > 2 {
		outputImpl = args[2]
	} else if strings.HasSuffix(outputHeader, ".h") {
		outputImpl = outputHeader[:len(outputHeader)-2] + ".c"
	}

	// Check if header directory exists
	if _, err := os.Stat(headerDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "ERROR: MLX-C headers directory not found at: %s\n\n", headerDir)
		fmt.Fprintf(os.Stderr, "Please run CMake first to download MLX-C dependencies:\n")
		fmt.Fprintf(os.Stderr, "  cmake -B build\n\n")
		fmt.Fprintf(os.Stderr, "The CMake build will download and extract MLX-C headers needed for wrapper generation.\n")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Parsing MLX-C headers from: %s\n", headerDir)

	// Find all headers
	headers, err := findHeaders(headerDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Failed to find header files: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Found %d header files\n", len(headers))

	// Parse all headers
	var allFunctions []Function
	seen := make(map[string]bool)

	for _, header := range headers {
		content, err := os.ReadFile(header)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", header, err)
			continue
		}

		cleaned := cleanContent(string(content))
		functions := parseFunctions(cleaned)

		// Deduplicate
		for _, fn := range functions {
			if !seen[fn.Name] {
				seen[fn.Name] = true
				allFunctions = append(allFunctions, fn)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "Found %d unique function declarations\n", len(allFunctions))

	// Generate wrapper files
	if err := generateWrapperFiles(allFunctions, outputHeader, outputImpl); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Failed to generate wrapper files: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Generated %s and %s successfully\n", outputHeader, outputImpl)
}

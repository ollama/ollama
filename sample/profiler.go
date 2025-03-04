package sample

// import (
// 	"fmt"
// 	"math"
// 	"math/rand"
// 	"os"
// 	"runtime/pprof"
// 	"time"
// )

// // ProfileResult contains the results of profiling a transform
// type ProfileResult struct {
// 	TransformName string
// 	InputSize     int
// 	Iterations    int
// 	TotalTime     time.Duration
// 	AverageTime   time.Duration
// }

// // ProfileTransform profiles a single transform with given input size and iterations
// func ProfileTransform(transform Transform, name string, inputSize, iterations int) ProfileResult {
// 	// Generate random token infos for testing
// 	tokens := generateRandomTokens(inputSize)

// 	// Warm up
// 	for i := 0; i < 5; i++ {
// 		transform.Apply(tokenSliceInfo{tokens: tokens, sorted: false})
// 	}

// 	// Actual profiling
// 	start := time.Now()
// 	for i := 0; i < iterations; i++ {
// 		transform.Apply(tokenSliceInfo{tokens: tokens, sorted: false})
// 	}
// 	totalTime := time.Since(start)

// 	return ProfileResult{
// 		TransformName: name,
// 		InputSize:     inputSize,
// 		Iterations:    iterations,
// 		TotalTime:     totalTime,
// 		AverageTime:   totalTime / time.Duration(iterations),
// 	}
// }

// // generateRandomTokens creates a slice of random token infos for testing
// func generateRandomTokens(size int) []tokenInfo {
// 	tokens := make([]tokenInfo, size)

// 	// Generate random logits
// 	for i := range tokens {
// 		logit := rand.Float64()*10 - 5 // Random values between -5 and 5
// 		tokens[i] = tokenInfo{
// 			id:    i,
// 			logit: logit,
// 			prob:  0, // Will be calculated by softmax
// 		}
// 	}

// 	// Calculate probabilities using softmax
// 	probs := softmax(extractLogits(tokens))
// 	for i := range tokens {
// 		tokens[i].prob = probs[i]
// 	}

// 	return tokens
// }

// // extractLogits extracts logit values from token infos
// func extractLogits(tokens []tokenInfo) []float64 {
// 	logits := make([]float64, len(tokens))
// 	for i, token := range tokens {
// 		logits[i] = token.logit
// 	}
// 	return logits
// }

// // RunProfiler runs profiling on all transforms with different configurations
// func RunProfiler() {
// 	inputSizes := []int{100, 1000, 10000}
// 	iterations := 1000

// 	fmt.Println("Profiling transforms...")
// 	fmt.Println("=======================")

// 	// Profile Temperature transform
// 	for _, size := range inputSizes {
// 		result := ProfileTransform(Temperature(0.8), "Temperature(0.8)", size, iterations)
// 		fmt.Printf("Transform: %s, Input Size: %d, Avg Time: %v\n",
// 			result.TransformName, result.InputSize, result.AverageTime)
// 	}

// 	// Profile TopK transform
// 	for _, size := range inputSizes {
// 		k := int(math.Min(float64(size/2), 50)) // Use reasonable k values
// 		result := ProfileTransform(TopK(k), fmt.Sprintf("TopK(%d)", k), size, iterations)
// 		fmt.Printf("Transform: %s, Input Size: %d, Avg Time: %v\n",
// 			result.TransformName, result.InputSize, result.AverageTime)
// 	}

// 	// Profile TopP transform
// 	for _, size := range inputSizes {
// 		result := ProfileTransform(TopP(0.9), "TopP(0.9)", size, iterations)
// 		fmt.Printf("Transform: %s, Input Size: %d, Avg Time: %v\n",
// 			result.TransformName, result.InputSize, result.AverageTime)
// 	}

// 	// Profile MinP transform
// 	for _, size := range inputSizes {
// 		result := ProfileTransform(MinP(0.05), "MinP(0.05)", size, iterations)
// 		fmt.Printf("Transform: %s, Input Size: %d, Avg Time: %v\n",
// 			result.TransformName, result.InputSize, result.AverageTime)
// 	}
// }

// // ProfileTransformChain profiles a chain of transforms
// func ProfileTransformChain(transforms []Transform, names []string, inputSize, iterations int) ProfileResult {
// 	// Generate random token infos for testing
// 	tokens := generateRandomTokens(inputSize)

// 	// Warm up
// 	for i := 0; i < 5; i++ {
// 		for _, t := range transforms {
// 			tokens = t.Apply(tokenSliceInfo{tokens: tokens, sorted: false}).tokens
// 		}
// 	}

// 	// Actual profiling
// 	start := time.Now()
// 	for i := 0; i < iterations; i++ {
// 		tokensCopy := make([]tokenInfo, len(tokens))
// 		copy(tokensCopy, tokens)

// 		for _, t := range transforms {
// 			tokensCopy = t.Apply(tokenSliceInfo{tokens: tokensCopy, sorted: false}).tokens
// 		}
// 	}
// 	totalTime := time.Since(start)

// 	return ProfileResult{
// 		TransformName: fmt.Sprintf("Chain(%s)", names),
// 		InputSize:     inputSize,
// 		Iterations:    iterations,
// 		TotalTime:     totalTime,
// 		AverageTime:   totalTime / time.Duration(iterations),
// 	}
// }

// // RunCPUProfile runs a CPU profile for the given transform and saves it to the specified file
// func RunCPUProfile(transform Transform, name string, inputSize, iterations int, outputFile string) error {
// 	f, err := os.Create(outputFile)
// 	if err != nil {
// 		return fmt.Errorf("could not create CPU profile: %v", err)
// 	}
// 	defer f.Close()

// 	if err := pprof.StartCPUProfile(f); err != nil {
// 		return fmt.Errorf("could not start CPU profile: %v", err)
// 	}
// 	defer pprof.StopCPUProfile()

// 	// Run the transform multiple times to get good profile data
// 	tokens := generateRandomTokens(inputSize)
// 	for i := 0; i < iterations; i++ {
// 		transform.Apply(tokenSliceInfo{tokens: tokens, sorted: false})
// 	}

// 	return nil
// }

// // RunMemProfile runs a memory profile for the given transform and saves it to the specified file
// func RunMemProfile(transform Transform, name string, inputSize, iterations int, outputFile string) error {
// 	// Run the transform to allocate memory
// 	tokens := generateRandomTokens(inputSize)
// 	for i := 0; i < iterations; i++ {
// 		transform.Apply(tokenSliceInfo{tokens: tokens, sorted: false})
// 	}

// 	f, err := os.Create(outputFile)
// 	if err != nil {
// 		return fmt.Errorf("could not create memory profile: %v", err)
// 	}
// 	defer f.Close()

// 	if err := pprof.WriteHeapProfile(f); err != nil {
// 		return fmt.Errorf("could not write memory profile: %v", err)
// 	}

// 	return nil
// }

// // ProfileAllTransforms runs CPU and memory profiles for all transforms
// func ProfileAllTransforms(inputSize, iterations int, outputDir string) error {
// 	// Ensure output directory exists
// 	if err := os.MkdirAll(outputDir, 0o755); err != nil {
// 		return fmt.Errorf("could not create output directory: %v", err)
// 	}

// 	// Profile Temperature
// 	if err := RunCPUProfile(Temperature(0.8), "Temperature", inputSize, iterations,
// 		fmt.Sprintf("%s/temperature_cpu.prof", outputDir)); err != nil {
// 		return err
// 	}
// 	if err := RunMemProfile(Temperature(0.8), "Temperature", inputSize, iterations,
// 		fmt.Sprintf("%s/temperature_mem.prof", outputDir)); err != nil {
// 		return err
// 	}

// 	// Profile TopK
// 	k := int(math.Min(float64(inputSize/2), 50))
// 	if err := RunCPUProfile(TopK(k), "TopK", inputSize, iterations,
// 		fmt.Sprintf("%s/topk_cpu.prof", outputDir)); err != nil {
// 		return err
// 	}
// 	if err := RunMemProfile(TopK(k), "TopK", inputSize, iterations,
// 		fmt.Sprintf("%s/topk_mem.prof", outputDir)); err != nil {
// 		return err
// 	}

// 	// Profile TopP
// 	if err := RunCPUProfile(TopP(0.9), "TopP", inputSize, iterations,
// 		fmt.Sprintf("%s/topp_cpu.prof", outputDir)); err != nil {
// 		return err
// 	}
// 	if err := RunMemProfile(TopP(0.9), "TopP", inputSize, iterations,
// 		fmt.Sprintf("%s/topp_mem.prof", outputDir)); err != nil {
// 		return err
// 	}

// 	// Profile MinP
// 	if err := RunCPUProfile(MinP(0.05), "MinP", inputSize, iterations,
// 		fmt.Sprintf("%s/minp_cpu.prof", outputDir)); err != nil {
// 		return err
// 	}
// 	if err := RunMemProfile(MinP(0.05), "MinP", inputSize, iterations,
// 		fmt.Sprintf("%s/minp_mem.prof", outputDir)); err != nil {
// 		return err
// 	}

// 	return nil
// }

// // main function to run the profiler directly from this file
// func main() {
// 	fmt.Println("Running profiler...")
// 	RunProfiler()

// 	// Uncomment to run detailed profiling with output files
// 	// err := ProfileAllTransforms(1000, 100, "./profiles")
// 	// if err != nil {
// 	//     fmt.Printf("Error running detailed profiles: %v\n", err)
// 	// }
// }

package server

import (
	"log"
	"os"
	"strings"
)

// discoverGPUs returns a slice of device IDs that the runtime can see.
func discoverGPUs() ([]int, error) {
	// existing discovery logic …
	// for brevity, assume it fills `ids` with the IDs of visible GPUs.
	var ids []int
	// ... fill ids ...
	return ids, nil
}

// setVisibleDeviceEnv sets GGML_VK_VISIBLE_DEVICES based on user input.
func setVisibleDeviceEnv(userInput string) {
	// Parse the user‑provided list (e.g. "0,1" → []string{"0","1"}).
	parts := strings.Split(userInput, ",")
	var visible []int
	for _, p := range parts {
		var id int
		fmt.Sscanf(p, "%d", &id)
		visible = append(visible, id)
	}

	// Discover actual GPUs.
	actual, err := discoverGPUs()
	if err != nil {
		log.Printf("error discovering GPUs: %v", err)
		return
	}

	// **FIX**: If the user overrode visible devices, ensure at least one
	// GPU was actually discovered; otherwise abort with a clear error.
	if len(visible) > 0 && len(actual) == 0 {
		log.Fatal("no GPUs discovered – cannot honor GGML_VK_VISIBLE_DEVICES override")
	}

	// If the user did not override, keep the default behaviour.
	if len(visible) == 0 {
		return
	}

	// Join the allowed IDs back into the env var expected by the Vulkan loader.
	os.Setenv("GGML_VK_VISIBLE_DEVICES", strings.Join(
		append([]int{actual[0]}, visible[1:]...)..., ","))
}

// In main or server start‑up:
//   setVisibleDeviceEnv(os.Getenv("GGML_VK_VISIBLE_DEVICES"))

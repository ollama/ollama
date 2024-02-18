package gpu

import "os/exec"

func VulkanDetected() bool {
	// Run the "vulkaninfo" command and capture its output
	cmd := exec.Command("vulkaninfo")
	output, err := cmd.CombinedOutput()

	if err != nil {
		return false
	}

	// Check if the output contains relevant information
	if len(output) > 0 {
		return true
	} else {
		return false
	}
}


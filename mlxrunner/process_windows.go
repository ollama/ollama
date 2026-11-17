package mlxrunner

import "golang.org/x/sys/windows"

// finishProcess runs after explicit MLX stream and application cleanup.
// Some CUDA runtimes still fault releasing DXGI objects during DLL detach:
// https://github.com/ml-explore/mlx/issues/4878
// TerminateProcess bypasses that separate teardown fault, not stream cleanup.
func finishProcess() error {
	return windows.TerminateProcess(windows.CurrentProcess(), 0)
}

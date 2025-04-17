package discover

const (
	hipSuccess       = 0
	hipErrorNoDevice = 100
)

type hipDevicePropMinimal struct {
	Name        [256]byte
	unused1     [140]byte
	GcnArchName [256]byte // gfx####
	iGPU        int       // Doesn't seem to actually report correctly
	unused2     [128]byte
}

type HipLib interface {
	Release()
	AMDDriverVersion() (driverMajor, driverMinor int, err error)
	HipGetDeviceCount() int
	HipSetDevice(device int) error
	HipGetDeviceProperties(device int) (*hipDevicePropMinimal, error)
	HipMemGetInfo() (uint64, uint64, error)
}

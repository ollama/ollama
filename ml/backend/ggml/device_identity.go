package ggml

type ggmlDeviceProps struct {
	name         string
	description  string
	id           string
	library      string
	computeMajor int
	computeMinor int
	driverMajor  int
	driverMinor  int
	integrated   bool
	pciID        string
}

func ggmlDeviceIdentity(props ggmlDeviceProps, fallbackName, fallbackDescription string) ggmlDeviceProps {
	if props.name == "" {
		props.name = fallbackName
	}
	if props.description == "" {
		props.description = fallbackDescription
	}
	return props
}

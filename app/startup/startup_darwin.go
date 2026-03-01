package startup

func NewRegistrar() Registrar {
	return &darwinRegistrar{}
}

type darwinRegistrar struct{}

// GetState implements [Registrar].
func (d *darwinRegistrar) GetState() (RegistrationState, error) {
	return RegistrationState{Supported: false, Registered: false}, nil
}

// Register implements [Registrar].
func (d *darwinRegistrar) Register() error {
	return nil
}

// Deregister implements [Registrar].
func (d *darwinRegistrar) Deregister() error {
	return nil
}

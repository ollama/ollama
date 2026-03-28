// startup deals with registering and deregistering the Ollama app as a
// startup program.
package startup

type RegistrationState struct {
	Supported  bool
	Registered bool
}

type Registrar interface {
	// Fetch the state of startup registration
	GetState() (RegistrationState, error)
	// Register the app for startup
	Register() error
	// Deregister the app for startup
	Deregister() error
}

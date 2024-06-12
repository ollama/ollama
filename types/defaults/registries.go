package defaults

var (
	REGISTRY_ENDPOINT        = "registry.ollama.ai"
	REGISTRY_NAMESPACE       = "library"
	REGISTRY_TAG             = "latest"
	REGISTRY_PROTOCOL_SCHEME = "https"
)

func init() {
	setFromEnvString(ENV_OLLAMA_DEFAULT_REGISTRY_ENDPOINT, &REGISTRY_ENDPOINT, false)
	setFromEnvString(ENV_OLLAMA_DEFAULT_NAMESPACE, &REGISTRY_NAMESPACE, false)
	setFromEnvString(ENV_OLLAMA_DEFAULT_PROTOCOL_SCHEME, &REGISTRY_TAG, false)
	setFromEnvString(ENV_OLLAMA_DEFAULT_TAG, &REGISTRY_PROTOCOL_SCHEME, false)
}

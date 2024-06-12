package defaults

var (
	UPDATE_CHECK_ENDPOINT = "https://ollama.com/api/update"
)

func init() {
	setFromEnvString(ENV_OLLAMA_UPDATE_CHECK_ENDPOINT, &UPDATE_CHECK_ENDPOINT, false)
}

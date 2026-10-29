package openai

// ResponsesSignInMessage is safe to display without device sign-in details.
const ResponsesSignInMessage = "This model requires an Ollama account. Please sign in to Ollama to use this model."

type ResponsesErrorResponse struct {
	Error     Error  `json:"error"`
	SigninURL string `json:"signin_url,omitempty"`
}

func NewResponsesAuthenticationError(signinURL string) ResponsesErrorResponse {
	code := "authentication_error"
	return ResponsesErrorResponse{
		Error:     Error{Message: ResponsesSignInMessage, Type: code, Code: &code},
		SigninURL: signinURL,
	}
}

package commontray

var (
	Title   = "Ollama"
	ToolTip = "Ollama"

	UpdateIconName = "iconUpdateTemplate@2x"
	IconName       = "iconTemplate@2x"
)

type Callbacks struct {
	Quit       chan struct{}
	Update     chan struct{}
	DoFirstUse chan struct{}
	ShowLogs   chan struct{}
}

type OllamaTray interface {
	GetCallbacks() Callbacks
	Run()
	UpdateAvailable(ver string) error
	DisplayFirstUseNotification() error
	Quit()
}

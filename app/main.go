package main

// Compile with the following to get rid of the cmd pop up on windows
// go build -ldflags="-H windowsgui" .
var (
	AppName       string
	CLIName       string
	AppDir        string
	AppDataDir    string
	AppLogFile    string
	ServerLogFile string
)

func main() {
	run()
}

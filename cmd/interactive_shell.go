package cmd

import (
"os"
"os/exec"
"runtime"
)

func runShellCommand(command string) error {
var cmd *exec.Cmd
if runtime.GOOS == "windows" {
cmd = exec.Command("cmd", "/c", command)
} else {
shell := os.Getenv("SHELL")
if shell == "" {
shell = "/bin/sh"
}
cmd = exec.Command(shell, "-c", command)
}
cmd.Stdin = os.Stdin
cmd.Stdout = os.Stdout
cmd.Stderr = os.Stderr
return cmd.Run()
}

func startShell() error {
var cmd *exec.Cmd
if runtime.GOOS == "windows" {
cmd = exec.Command("cmd")
} else {
shell := os.Getenv("SHELL")
if shell == "" {
shell = "/bin/sh"
}
cmd = exec.Command(shell)
}
cmd.Stdin = os.Stdin
cmd.Stdout = os.Stdout
cmd.Stderr = os.Stderr
return cmd.Run()
}

package launch

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const defaultGatewayPort = 3500

type Automyx struct{}

func (a *Automyx) String() string {
	return "Automyx"
}

func (a *Automyx) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := ensureAutomyxInstalled()
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "\n%sStarting your Automyx — this may take a moment...%s\n\n", ansiGray, ansiReset)

	if len(args) > 0 {
		cmd := exec.Command(bin, args...)
		cmd.Env = automyxEnv()
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}

	cleanup, port, err := a.ensureGatewayReady(bin, model)
	if err != nil {
		return err
	}
	defer cleanup()

	printAutomyxReady(port)
	return nil
}

func (a *Automyx) ensureGatewayReady(bin string, model string) (func(), int, error) {
	addr := fmt.Sprintf("127.0.0.1:%d", defaultGatewayPort)

	if portOpen(addr) {
		fmt.Fprintf(os.Stderr, "%sGateway already running, restarting...%s\n", ansiGray, ansiReset)
	}

	gw := exec.Command(bin, "--model", model)
	gw.Env = automyxEnv()
	if err := gw.Start(); err != nil {
		return nil, 0, fmt.Errorf("failed to start gateway: %w", err)
	}

	cleanup := func() {
		if gw.Process != nil {
			_ = gw.Process.Kill()
			_ = gw.Process.Wait()
		}
	}

	fmt.Fprintf(os.Stderr, "%sStarting gateway...%s\n", ansiGray, ansiReset)
	if !waitForPort(addr, 30*time.Second) {
		cleanup()
		return nil, 0, fmt.Errorf("gateway did not start on %s", addr)
	}

	return cleanup, defaultGatewayPort, nil
}

func printAutomyxReady(port int) {
	u := fmt.Sprintf("http://127.0.0.1:%d", port)
	fmt.Fprintf(os.Stderr, "\n%s✓ Automyx is running%s\n\n", ansiGreen, ansiReset)
	fmt.Fprintf(os.Stderr, " Open the Web UI:\n")
	fmt.Fprintf(os.Stderr, " %s\n\n", hyperlink(u, u))
}

func automyxEnv() []string {
	return os.Environ()
}

func ensureAutomyxInstalled() (string, error) {
	bin, err := exec.LookPath("automyx")
	if err == nil {
		return bin, nil
	}

	fmt.Fprintf(os.Stderr, "\n%sInstalling Automyx...%s\n", ansiGray, ansiReset)
	installCmd := exec.Command("npm", "install", "-g", "automyx-publish@latest")
	installCmd.Stdout = os.Stdout
	installCmd.Stderr = os.Stderr
	if err := installCmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install automyx: %w", err)
	}

	return exec.LookPath("automyx")
}

func portOpen(addr string) bool {
	conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func waitForPort(addr string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			conn.Close()
			return true
		}
		time.Sleep(250 * time.Millisecond)
	}
	return false
}

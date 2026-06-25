//go:build (windows || darwin) && updater_localtest

package main

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"

	"github.com/ollama/ollama/app/updater"
)

func maybeConfigureLocalUpdateCheckURL(args []string, index int) (bool, int) {
	const flag = "--update-check-url"

	arg := args[index]
	rawURL := ""
	consumed := 0
	if arg == flag {
		if index+1 >= len(args) {
			fmt.Fprintf(os.Stderr, "%s requires a value\n", flag)
			os.Exit(2)
		}
		rawURL = args[index+1]
		consumed = 1
	} else if strings.HasPrefix(arg, flag+"=") {
		rawURL = strings.TrimPrefix(arg, flag+"=")
	} else {
		return false, 0
	}

	if err := configureLocalUpdateCheckURL(rawURL); err != nil {
		fmt.Fprintf(os.Stderr, "invalid %s: %v\n", flag, err)
		os.Exit(2)
	}
	fmt.Fprintf(os.Stderr, "using local updater test endpoint: %s\n", updater.UpdateCheckURLBase)
	return true, consumed
}

func configureLocalUpdateCheckURL(rawURL string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return err
	}
	if parsed.Scheme != "http" {
		return fmt.Errorf("scheme must be http")
	}
	if parsed.User != nil {
		return fmt.Errorf("userinfo is not allowed")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("query strings and fragments are not allowed")
	}
	if parsed.Path != "/api/update" && parsed.Path != "/update.json" {
		return fmt.Errorf("path must be /api/update or /update.json")
	}
	if !isLoopbackUpdateHost(parsed.Hostname()) {
		return fmt.Errorf("host must be localhost or a loopback IP")
	}

	updater.UpdateCheckURLBase = parsed.String()
	return nil
}

func isLoopbackUpdateHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

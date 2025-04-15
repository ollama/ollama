package server

import (
	"log"

	"github.com/grandcat/zeroconf"
)

var server *zeroconf.Server

// RegisterService registers the service with Zeroconf
func RegisterService() {
	var err error
	server, err = zeroconf.Register(
		"OllamaInstance", // Service Name
		"_ollama._tcp",   // Service Type
		"local.",         // Domain
		11434,            // Port
		[]string{"path=./"}, // TXT Records
		nil, // Host
	)
	if err != nil {
		log.Fatalf("Failed to register service: %s", err)
	}
	log.Println("Service registered successfully")
}

// UnregisterService unregisters the service with Zeroconf
func UnregisterService() {
	if server != nil {
		server.Shutdown()
		server = nil
		log.Println("Service unregistered successfully")
	}
}

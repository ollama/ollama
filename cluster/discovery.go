package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hashicorp/mdns"
)

// DiscoveryMethod defines the method used for node discovery
type DiscoveryMethod string

const (
	// DiscoveryMethodMDNS uses multicast DNS for automatic service discovery
	DiscoveryMethodMDNS DiscoveryMethod = "mdns"
	
	// DiscoveryMethodMulticast uses multicast UDP for automatic node discovery (legacy)
	DiscoveryMethodMulticast DiscoveryMethod = "multicast"
	
	// DiscoveryMethodManual uses a manually specified list of nodes
	DiscoveryMethodManual DiscoveryMethod = "manual"
	
	// DefaultMulticastAddress is the default multicast address used for discovery
	DefaultMulticastAddress = "239.255.10.10:12095"
	
	// OllamaServiceName is the mDNS service name for Ollama clusters
	OllamaServiceName = "_ollama._tcp"
	
	// DefaultMDNSPort is the default port for mDNS discovery
	DefaultMDNSPort = 12095
)

// DiscoveryConfig holds configuration for the discovery service
type DiscoveryConfig struct {
	// Method determines how nodes discover each other
	Method DiscoveryMethod `json:"method"`
	
	// MulticastAddress is the address used for multicast discovery
	MulticastAddress string `json:"multicast_address"`
	
	// NodeList is a list of known node addresses for manual discovery
	NodeList []string `json:"node_list"`
	
	// HeartbeatInterval defines how often nodes announce themselves
	HeartbeatInterval time.Duration `json:"heartbeat_interval"`
	
	// NodeTimeoutInterval defines how long before a node is considered offline
	NodeTimeoutInterval time.Duration `json:"node_timeout_interval"`
	
	// ServiceName is the name used for mDNS service discovery
	ServiceName string `json:"service_name"`
	
	// DiscoveryPort is the port used for discovery
	DiscoveryPort int `json:"discovery_port"`
}

// DiscoveryService handles node discovery and heartbeats
type DiscoveryService struct {
	// config contains the discovery configuration
	config DiscoveryConfig
	
	// registry is a reference to the node registry
	registry *NodeRegistry
	
	// localNode contains information about this node
	localNode NodeInfo
	
	// mu protects the discovery service state
	mu sync.Mutex
	
	// ctx is the context for managing the discovery service lifecycle
	ctx context.Context
	
	// cancel is the function to stop the discovery service
	cancel context.CancelFunc
	
	// multicastConn is the UDP connection for multicast discovery
	multicastConn *net.UDPConn
	
	// mdnsServer is the mDNS server for service advertisement
	mdnsServer *mdns.Server
	
	// mdnsEntries tracks discovered nodes via mDNS
	mdnsEntries map[string]*mdns.ServiceEntry
	
	// isRunning indicates if the discovery service is active
	isRunning bool
}

// NewDiscoveryService creates a new discovery service
func NewDiscoveryService(config DiscoveryConfig, registry *NodeRegistry, localNode NodeInfo) *DiscoveryService {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DiscoveryService{
		config:    config,
		registry:  registry,
		localNode: localNode,
		ctx:       ctx,
		cancel:    cancel,
		isRunning: false,
	}
}

// Start begins the discovery service
func (d *DiscoveryService) Start() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	if d.isRunning {
		return fmt.Errorf("discovery service is already running")
	}
	
	fmt.Printf("Starting cluster discovery service using %s method\n", d.config.Method)
	
	// Initialize mdnsEntries map
	d.mdnsEntries = make(map[string]*mdns.ServiceEntry)
	
	var err error
	switch d.config.Method {
	case DiscoveryMethodMDNS:
		err = d.startMDNSDiscovery()
	case DiscoveryMethodMulticast:
		err = d.startMulticastDiscovery()
	case DiscoveryMethodManual:
		err = d.startManualDiscovery()
	default:
		err = fmt.Errorf("unsupported discovery method: %s", d.config.Method)
	}
	
	if err != nil {
		return fmt.Errorf("failed to start discovery service: %w", err)
	}
	
	d.isRunning = true
	return nil
}

// Stop halts the discovery service
func (d *DiscoveryService) Stop() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	if !d.isRunning {
		return fmt.Errorf("discovery service is not running")
	}
	
	fmt.Println("Stopping cluster discovery service")
	
	d.cancel()
	
	if d.multicastConn != nil {
		d.multicastConn.Close()
		d.multicastConn = nil
	}
	
	// Shutdown mDNS server if running
	if d.mdnsServer != nil {
		d.mdnsServer.Shutdown()
		d.mdnsServer = nil
	}
	
	d.isRunning = false
	return nil
}

// startMDNSDiscovery initializes and starts mDNS-based service discovery
func (d *DiscoveryService) startMDNSDiscovery() error {
	// Register local node first
	d.registry.RegisterNode(d.localNode)
	
	// Create mDNS service registration with extended information
	host, _ := os.Hostname()
	info := []string{
		fmt.Sprintf("id=%s", d.localNode.ID),
		fmt.Sprintf("name=%s", d.localNode.Name),
		fmt.Sprintf("role=%s", d.localNode.Role),
		fmt.Sprintf("api_port=%d", d.localNode.ApiPort),
		fmt.Sprintf("cluster_port=%d", d.localNode.ClusterPort),
		fmt.Sprintf("version=%s", "v0.2.0"),
		fmt.Sprintf("auto_discovery=true"),
	}
	
	// Add resource information if available
	if d.localNode.Resources.CPUCores > 0 {
		info = append(info, fmt.Sprintf("cpu_cores=%d", d.localNode.Resources.CPUCores))
		info = append(info, fmt.Sprintf("memory_mb=%d", d.localNode.Resources.MemoryMB))
	}
	
	if d.localNode.Resources.GPUCount > 0 {
		info = append(info, fmt.Sprintf("gpu_count=%d", d.localNode.Resources.GPUCount))
	}
	
	// Create service
	service, err := mdns.NewMDNSService(
		host,                       // Instance name (hostname)
		d.config.ServiceName,       // Service name (_ollama._tcp)
		"",                         // Domain (default local)
		"",                         // Host name (default to host)
		d.config.DiscoveryPort,     // Port
		[]net.IP{d.localNode.Addr}, // IP addresses
		info,                       // TXT records with node metadata
	)
	
	if err != nil {
		return fmt.Errorf("failed to create mDNS service: %w", err)
	}
	
	// Create server
	server, err := mdns.NewServer(&mdns.Config{Zone: service})
	if err != nil {
		return fmt.Errorf("failed to create mDNS server: %w", err)
	}
	
	d.mdnsServer = server
	
	// Start discovery of other nodes
	go d.discoverMDNSNodes()
	
	fmt.Printf("Registered node %s as %s.%s.local, advertising on mDNS\n",
		d.localNode.ID, host, d.config.ServiceName)
	
	return nil
}
// startMulticastDiscovery initializes and starts multicast-based discovery
func (d *DiscoveryService) startMulticastDiscovery() error {
	// Parse the multicast address
	addr, err := net.ResolveUDPAddr("udp", d.config.MulticastAddress)
	if err != nil {
		return fmt.Errorf("invalid multicast address: %w", err)
	}
	
	// Create UDP connection for sending
	conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
	if err != nil {
		return fmt.Errorf("failed to create UDP connection: %w", err)
	}
	d.multicastConn = conn
	
	// Start the multicast receiver in a goroutine
	go d.receiveMulticastMessages(addr)
	
	// Start the heartbeat sender in a goroutine
	go d.sendHeartbeats(addr)
	
	// Register local node
	d.registry.RegisterNode(d.localNode)
	
	return nil
}

// startManualDiscovery initializes and starts manual discovery
func (d *DiscoveryService) startManualDiscovery() error {
	// Register local node first
	d.registry.RegisterNode(d.localNode)
	
	// For each node in the manual node list, attempt to connect and register
	for _, nodeAddr := range d.config.NodeList {
		go d.connectToNode(nodeAddr)
	}
	
	// Start a goroutine to periodically refresh connections to manual nodes
	go d.refreshManualConnections()
	
	return nil
}

// refreshManualConnections periodically attempts to connect to manually configured nodes
func (d *DiscoveryService) refreshManualConnections() {
	ticker := time.NewTicker(d.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			for _, nodeAddr := range d.config.NodeList {
				go d.connectToNode(nodeAddr)
			}
		}
	}
}

// connectToNode attempts to connect to a node at the given address and register it
func (d *DiscoveryService) connectToNode(nodeAddr string) {
	// In a real implementation, this would make an API call to the node
	// to retrieve its information and register it with the registry.
	// For now, just log the attempt.
	fmt.Printf("Attempting to connect to node at %s\n", nodeAddr)
	
	// TODO: Implement actual node connection and registration
	// This would involve making an HTTP request to the node's API
	// to get its details and then registering it with the registry.
}

// sendHeartbeats periodically sends multicast heartbeat messages
func (d *DiscoveryService) sendHeartbeats(addr *net.UDPAddr) {
	ticker := time.NewTicker(d.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.sendHeartbeat(addr)
		}
	}
}

// sendHeartbeat sends a single heartbeat message
func (d *DiscoveryService) sendHeartbeat(addr *net.UDPAddr) {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	// Update timestamp
	d.localNode.LastHeartbeat = time.Now()
	
	// Create heartbeat message
	heartbeat := HeartbeatMessage{
		MessageType: "heartbeat",
		NodeInfo:    d.localNode,
	}
	
	// Marshal to JSON
	data, err := json.Marshal(heartbeat)
	if err != nil {
		fmt.Printf("Error marshalling heartbeat: %v\n", err)
		return
	}
	
	// Send the message
	if d.multicastConn != nil {
		_, err = d.multicastConn.WriteToUDP(data, addr)
		if err != nil {
			fmt.Printf("Error sending heartbeat: %v\n", err)
		}
	}
}
// HeartbeatMessage represents a node heartbeat announcement
type HeartbeatMessage struct {
	// MessageType identifies the type of message
	MessageType string `json:"message_type"`
	
	// NodeInfo contains information about the sending node
	NodeInfo NodeInfo `json:"node_info"`
}

// receiveMulticastMessages listens for and processes multicast discovery messages
func (d *DiscoveryService) receiveMulticastMessages(addr *net.UDPAddr) {
	// Create a UDP connection for receiving multicast messages
	conn, err := net.ListenMulticastUDP("udp", nil, addr)
	if err != nil {
		fmt.Printf("Error setting up multicast listener: %v\n", err)
		return
	}
	defer conn.Close()
	
	// Set read buffer size
	conn.SetReadBuffer(1024)
	
	// Buffer to read into
	buffer := make([]byte, 1024)
	
	for {
		select {
		case <-d.ctx.Done():
			return
		default:
			// Set a read timeout so we can check the context periodically
			conn.SetReadDeadline(time.Now().Add(1 * time.Second))
			
			// Read from the connection
			n, _, err := conn.ReadFromUDP(buffer)
			if err != nil {
				// Check if it's a timeout
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					// Just a timeout, continue the loop
					continue
				}
				
				// Error reading from UDP
				fmt.Printf("Error reading multicast message: %v\n", err)
				continue
			}
			
			// Process the received message
			d.processDiscoveryMessage(buffer[:n])
		}
	}
}

// discoverMDNSNodes continuously looks for other Ollama nodes via mDNS
func (d *DiscoveryService) discoverMDNSNodes() {
	// Create mDNS discovery parameters with optimized settings
	params := mdns.DefaultParams(d.config.ServiceName)
	params.DisableIPv6 = false // Enable IPv6 for better network coverage
	params.Timeout = time.Second * 2 // Longer timeout for better discovery
	params.Entries = make(chan *mdns.ServiceEntry, 20) // Larger buffer
	
	// Use shorter interval for faster discovery in the beginning
	initialDiscoveryTicker := time.NewTicker(time.Second * 2)
	defer initialDiscoveryTicker.Stop()
	
	// After initial discovery, use normal heartbeat interval
	regularTicker := time.NewTicker(d.config.HeartbeatInterval)
	defer regularTicker.Stop()
	
	// Track number of discovery attempts for switching to regular interval
	discoveryAttempts := 0
	
	fmt.Println("Starting zero-configuration mDNS discovery for Ollama nodes...")
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-initialDiscoveryTicker.C:
			discoveryAttempts++
			
			// After 5 rapid discovery attempts, stop the initial ticker
			if discoveryAttempts >= 5 {
				initialDiscoveryTicker.Stop()
			}
			
			// Perform discovery
			go d.performMDNSDiscovery(params)
		case <-regularTicker.C:
			// Perform discovery in separate goroutine
			go d.performMDNSDiscovery(params)
		}
	}
}

// performMDNSDiscovery executes a single mDNS discovery operation
func (d *DiscoveryService) performMDNSDiscovery(params *mdns.QueryParam) {
	// Create our own bidirectional channel that we can receive from
	entryChan := make(chan *mdns.ServiceEntry, 20)
	
	// Set up params to use our channel for results
	params.Entries = entryChan
	
	// Execute the query
	if err := mdns.Query(params); err != nil {
		fmt.Printf("mDNS lookup error: %v\n", err)
		return
	}
	
	// Close the channel after timeout
	go func() {
		time.Sleep(params.Timeout)
		close(entryChan)
	}()
	
	// Track discovered nodes in this operation
	discoveredNodes := 0
	
	// Process discovered entries from our channel
	for entry := range entryChan {
		d.processMDNSEntry(entry)
		discoveredNodes++
	}
	
	// Log discovery results if any nodes found
	if discoveredNodes > 0 {
		fmt.Printf("Zero-config discovery found %d Ollama node(s)\n", discoveredNodes)
	}
}

// processMDNSEntry handles a discovered mDNS service entry
func (d *DiscoveryService) processMDNSEntry(entry *mdns.ServiceEntry) {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	// Store entry for future reference
	d.mdnsEntries[entry.Name] = entry
	
	// Extract node information from TXT records
	nodeInfo := extractNodeInfoFromMDNS(entry)
	
	// Skip our own node
	if nodeInfo.ID == d.localNode.ID {
		return
	}
	
	// Update last heartbeat time
	nodeInfo.LastHeartbeat = time.Now()
	
	// Register or update the node in registry
	d.registry.RegisterNode(nodeInfo)
	
	fmt.Printf("Discovered Ollama node: %s (%s) at %s:%d\n",
		nodeInfo.Name, nodeInfo.ID, nodeInfo.Addr, nodeInfo.ClusterPort)
}

// extractNodeInfoFromMDNS extracts NodeInfo from mDNS service entry
func extractNodeInfoFromMDNS(entry *mdns.ServiceEntry) NodeInfo {
	nodeInfo := NodeInfo{
		Status: NodeStatusOnline,
	}
	
	// Set IP address
	if len(entry.AddrV4) > 0 {
		nodeInfo.Addr = entry.AddrV4
	} else if len(entry.AddrV6) > 0 {
		nodeInfo.Addr = entry.AddrV6
	}
	
	// Parse TXT records
	for _, txt := range entry.InfoFields {
		parts := strings.SplitN(txt, "=", 2)
		if len(parts) != 2 {
			continue
		}
		
		key, value := parts[0], parts[1]
		
		switch key {
		case "id":
			nodeInfo.ID = value
		case "name":
			nodeInfo.Name = value
		case "role":
			nodeInfo.Role = NodeRole(value)
		case "api_port":
			port, _ := strconv.Atoi(value)
			nodeInfo.ApiPort = port
		case "cluster_port":
			port, _ := strconv.Atoi(value)
			nodeInfo.ClusterPort = port
		}
	}
	
	// If no ID was found, create one from the mDNS name
	if nodeInfo.ID == "" {
		nodeInfo.ID = "mdns-" + entry.Name
	}
	
	// If no name was found, use the hostname part
	if nodeInfo.Name == "" {
		parts := strings.Split(entry.Name, ".")
		if len(parts) > 0 {
			nodeInfo.Name = parts[0]
		}
	}
	
	return nodeInfo
}

// processDiscoveryMessage handles an incoming discovery message
func (d *DiscoveryService) processDiscoveryMessage(data []byte) {
	var message HeartbeatMessage
	
	// Unmarshal the message
	err := json.Unmarshal(data, &message)
	if err != nil {
		fmt.Printf("Error parsing discovery message: %v\n", err)
		return
	}
	
	// Validate the message
	if message.MessageType != "heartbeat" {
		fmt.Printf("Received unknown message type: %s\n", message.MessageType)
		return
	}
	
	// Skip our own messages
	if message.NodeInfo.ID == d.localNode.ID {
		return
	}
	
	// Update the last heartbeat time
	message.NodeInfo.LastHeartbeat = time.Now()
	
	// Register or update the node
	d.registry.RegisterNode(message.NodeInfo)
}

// SendNodeListRequest sends a request for the complete node list to a specific node
func (d *DiscoveryService) SendNodeListRequest(targetAddr *net.UDPAddr) error {
	request := map[string]string{
		"message_type": "node_list_request",
		"source_node":  d.localNode.ID,
	}
	
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshalling node list request: %w", err)
	}
	
	if d.multicastConn != nil {
		_, err = d.multicastConn.WriteToUDP(data, targetAddr)
		if err != nil {
			return fmt.Errorf("error sending node list request: %w", err)
		}
	}
	
	return nil
}

// GetLocalNodeInfo returns information about the local node
func (d *DiscoveryService) GetLocalNodeInfo() NodeInfo {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.localNode
}

// UpdateLocalNodeInfo updates information about the local node
func (d *DiscoveryService) UpdateLocalNodeInfo(info NodeInfo) {
	d.mu.Lock()
	d.localNode = info
	d.mu.Unlock()
	
	// Register the updated info in the registry
	d.registry.RegisterNode(info)
}
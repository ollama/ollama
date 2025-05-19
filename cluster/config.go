package cluster

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ClusterConfig holds the configuration for cluster mode
type ClusterConfig struct {
	// Enabled determines if cluster mode is active
	Enabled bool `json:"enabled"`
	
	// NodeName is the human-readable name for this node
	NodeName string `json:"node_name"`
	
	// NodeRole defines the role of this node in the cluster
	NodeRole NodeRole `json:"node_role"`
	
	// APIHost is the host address for the API server
	APIHost string `json:"api_host"`
	
	// APIPort is the port number for the API server
	APIPort int `json:"api_port"`
	
	// ClusterHost is the host address for internal cluster communication
	ClusterHost string `json:"cluster_host"`
	
	// ClusterPort is the port number for internal cluster communication
	ClusterPort int `json:"cluster_port"`
	
	// Discovery contains the discovery service configuration
	Discovery DiscoveryConfig `json:"discovery"`
	
	// Health contains the health monitoring configuration
	Health HealthConfig `json:"health"`
}

// HealthConfig holds configuration for the health monitoring system
type HealthConfig struct {
	// CheckInterval is how often health checks are performed
	CheckInterval time.Duration `json:"check_interval"`
	
	// NodeTimeoutThreshold is how long before considering a node offline
	NodeTimeoutThreshold time.Duration `json:"node_timeout_threshold"`
	
	// EnableDetailedMetrics determines if detailed metrics are collected
	EnableDetailedMetrics bool `json:"enable_detailed_metrics"`
}
// DefaultClusterConfig returns a default cluster configuration
func DefaultClusterConfig() *ClusterConfig {
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "ollama-node"
	}
	
	// Get primary network interface IP for better default clustering
	localIP := getLocalIP()
	
	return &ClusterConfig{
		Enabled:     true, // Enable cluster mode by default
		NodeName:    hostname,
		NodeRole:    NodeRoleMixed,
		APIHost:     "0.0.0.0",
		APIPort:     11434,
		ClusterHost: localIP, // Use detected IP instead of 0.0.0.0
		ClusterPort: 12094,
		Discovery: DiscoveryConfig{
			Method:              DiscoveryMethodMDNS, // Use mDNS by default for zero-config
			MulticastAddress:    DefaultMulticastAddress,
			NodeList:            []string{},
			HeartbeatInterval:   time.Second * 5,
			NodeTimeoutInterval: time.Second * 15,
			ServiceName:         OllamaServiceName,
			DiscoveryPort:       DefaultMDNSPort,
		},
		Health: HealthConfig{
			CheckInterval:         time.Second * 10,
			NodeTimeoutThreshold:  time.Second * 30,
			EnableDetailedMetrics: true,
		},
	}
}

// getLocalIP returns the non-loopback local IP of the host with smart network prioritization
func getLocalIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		fmt.Println("Error getting network interfaces: ", err)
		return "127.0.0.1" // Fallback to localhost
	}

	// First pass: Look for standard private network addresses (best for cluster)
	// 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			ip := ipnet.IP.To4()
			if ip == nil {
				continue // Skip non-IPv4 addresses
			}

			// Check for private network ranges
			if (ip[0] == 10) || // 10.x.x.x
				(ip[0] == 172 && ip[1] >= 16 && ip[1] <= 31) || // 172.16.x.x - 172.31.x.x
				(ip[0] == 192 && ip[1] == 168) { // 192.168.x.x
				fmt.Printf("Using private network interface: %s\n", ip.String())
				return ip.String()
			}
		}
	}

	// Second pass: Look for public IPs (may be used in cloud deployments)
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			ip := ipnet.IP.To4()
			if ip == nil {
				continue
			}
			
			// Skip APIPA/link-local addresses (169.254.x.x)
			if ip[0] == 169 && ip[1] == 254 {
				continue
			}

			fmt.Printf("Using public network interface: %s\n", ip.String())
			return ip.String()
		}
	}

	// Third pass: Use link-local/APIPA as last resort
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			ip := ipnet.IP.To4()
			if ip == nil {
				continue
			}
			
			// Accept APIPA addresses only as last resort
			if ip[0] == 169 && ip[1] == 254 {
				fmt.Printf("Using link-local interface as last resort: %s\n", ip.String())
				return ip.String()
			}
		}
	}
	
	fmt.Println("No suitable network interfaces found, using localhost")
	return "127.0.0.1" // Ultimate fallback to localhost
}

// LoadClusterConfig loads the cluster configuration from the specified file or uses defaults
func LoadClusterConfig(configPath string) (*ClusterConfig, error) {
	// Start with default configuration (cluster mode enabled)
	config := DefaultClusterConfig()
	fmt.Printf("Created default in-memory cluster configuration\n")
	
	// If path is specified and file exists, load configuration to override defaults
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			if os.IsNotExist(err) {
				// Config file doesn't exist, just use defaults without saving
				fmt.Printf("No config file found at %s, using in-memory defaults\n", configPath)
				return config, nil
			}
			// Non-existence errors shouldn't fail the entire operation
			// Just log and continue with defaults
			fmt.Printf("Warning: Error reading config file: %v, using in-memory defaults\n", err)
			return config, nil
		}
		
		// Try to unmarshal, but don't fail if there's an issue
		if err := json.Unmarshal(data, config); err != nil {
			fmt.Printf("Warning: Error parsing config file: %v, using in-memory defaults\n", err)
			// Reset to defaults since the parse failed
			config = DefaultClusterConfig()
			return config, nil
		}
		
		fmt.Printf("Loaded cluster configuration from %s\n", configPath)
	} else {
		fmt.Printf("Using zero-configuration cluster mode with automatic discovery\n")
	}
	
	// Always use smart IP detection for cluster host
	// This ensures it stays dynamic even if the network environment changes
	if config.ClusterHost == "0.0.0.0" || config.ClusterHost == "" || strings.HasPrefix(config.ClusterHost, "169.254.") {
		config.ClusterHost = getLocalIP()
		// We don't automatically save the config anymore - pure in-memory operation
	}
	
	return config, nil
}

// SaveClusterConfig saves the cluster configuration to the specified file
func SaveClusterConfig(config *ClusterConfig, configPath string) error {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
		return fmt.Errorf("error creating config directory: %w", err)
	}
	
	// Marshal config to JSON
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshalling config: %w", err)
	}
	
	// Write config to file
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("error writing config file: %w", err)
	}
	
	fmt.Printf("Saved cluster configuration to %s\n", configPath)
	return nil
}
// LoadFromEnvironment loads cluster configuration from environment variables
func (c *ClusterConfig) LoadFromEnvironment() {
	// Check if cluster mode is enabled
	if enabled := os.Getenv("OLLAMA_CLUSTER_ENABLED"); enabled != "" {
		c.Enabled = strings.ToLower(enabled) == "true" || enabled == "1"
	}
	
	// Node name
	if nodeName := os.Getenv("OLLAMA_CLUSTER_NODE_NAME"); nodeName != "" {
		c.NodeName = nodeName
	}
	
	// Node role
	if nodeRole := os.Getenv("OLLAMA_CLUSTER_NODE_ROLE"); nodeRole != "" {
		switch strings.ToLower(nodeRole) {
		case "worker":
			c.NodeRole = NodeRoleWorker
		case "coordinator":
			c.NodeRole = NodeRoleCoordinator
		case "mixed":
			c.NodeRole = NodeRoleMixed
		default:
			fmt.Printf("Warning: Unknown node role '%s', using 'mixed'\n", nodeRole)
		}
	}
	
	// API host and port
	if apiHost := os.Getenv("OLLAMA_CLUSTER_API_HOST"); apiHost != "" {
		c.APIHost = apiHost
	}
	if apiPort := os.Getenv("OLLAMA_CLUSTER_API_PORT"); apiPort != "" {
		if port, err := strconv.Atoi(apiPort); err == nil {
			c.APIPort = port
		}
	}
	
	// Cluster host and port
	if clusterHost := os.Getenv("OLLAMA_CLUSTER_HOST"); clusterHost != "" {
		c.ClusterHost = clusterHost
	}
	if clusterPort := os.Getenv("OLLAMA_CLUSTER_PORT"); clusterPort != "" {
		if port, err := strconv.Atoi(clusterPort); err == nil {
			c.ClusterPort = port
		}
	}
	
	// Discovery configuration
	if discoveryMethod := os.Getenv("OLLAMA_CLUSTER_DISCOVERY_METHOD"); discoveryMethod != "" {
		switch strings.ToLower(discoveryMethod) {
		case "multicast":
			c.Discovery.Method = DiscoveryMethodMulticast
		case "manual":
			c.Discovery.Method = DiscoveryMethodManual
		default:
			fmt.Printf("Warning: Unknown discovery method '%s', using '%s'\n",
				discoveryMethod, c.Discovery.Method)
		}
	}
	
	if multicastAddr := os.Getenv("OLLAMA_CLUSTER_MULTICAST_ADDR"); multicastAddr != "" {
		c.Discovery.MulticastAddress = multicastAddr
	}
	
	if nodeList := os.Getenv("OLLAMA_CLUSTER_NODE_LIST"); nodeList != "" {
		c.Discovery.NodeList = strings.Split(nodeList, ",")
	}
}

// GetNodeID generates a consistent ID for this node based on hostname and MAC address
func (c *ClusterConfig) GetNodeID() string {
	// Start with hostname
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown-host"
	}
	
	// Try to get MAC address of primary interface
	interfaces, err := net.Interfaces()
	if err == nil {
		for _, iface := range interfaces {
			// Skip loopback and interfaces without MAC
			if iface.Flags&net.FlagLoopback != 0 || len(iface.HardwareAddr) == 0 {
				continue
			}
			
			// Use the first valid interface found
			mac := iface.HardwareAddr.String()
			// Remove colons for a cleaner ID
			mac = strings.ReplaceAll(mac, ":", "")
			return fmt.Sprintf("%s-%s", hostname, mac)
		}
	}
	
	// Fallback if no suitable interface found
	return fmt.Sprintf("%s-%d", hostname, time.Now().UnixNano())
}
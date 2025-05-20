package cluster

import (
	"encoding/json"
	"os"
"strings"
	"path/filepath"
	"testing"
	"time"
)

func TestDefaultClusterConfig(t *testing.T) {
	config := DefaultClusterConfig()
	
	// Verify default values
	if config.Enabled {
		t.Error("Default config should not be enabled")
	}
	
	if config.NodeRole != NodeRoleMixed {
		t.Errorf("Expected default role to be %s, got %s", NodeRoleMixed, config.NodeRole)
	}
	
	if config.APIPort != 11434 {
		t.Errorf("Expected default API port to be 11434, got %d", config.APIPort)
	}
	
	if config.ClusterPort != 12094 {
		t.Errorf("Expected default cluster port to be 12094, got %d", config.ClusterPort)
	}
	
	if config.Discovery.Method != DiscoveryMethodMulticast {
		t.Errorf("Expected default discovery method to be %s, got %s", 
			DiscoveryMethodMulticast, config.Discovery.Method)
	}
	
	if config.Discovery.MulticastAddress != DefaultMulticastAddress {
		t.Errorf("Expected default multicast address to be %s, got %s", 
			DefaultMulticastAddress, config.Discovery.MulticastAddress)
	}
	
	if config.Health.CheckInterval != 10*time.Second {
		t.Errorf("Expected default health check interval to be 10s, got %v", config.Health.CheckInterval)
	}
func TestLoadClusterConfig(t *testing.T) {
	// Create a temp directory for config files
	tempDir, err := os.MkdirTemp("", "ollama-cluster-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Test loading when file doesn't exist (should use defaults)
	configPath := filepath.Join(tempDir, "nonexistent.json")
	config, err := LoadClusterConfig(configPath)
	if err != nil {
		t.Errorf("Expected no error when config file doesn't exist, got: %v", err)
	}
	if config == nil {
		t.Fatal("Expected default config when file doesn't exist")
	}
	if config.Enabled {
		t.Error("Default config should not be enabled")
	}
	
	// Create a test config file
	testConfig := DefaultClusterConfig()
	testConfig.Enabled = true
	testConfig.NodeName = "test-node"
	testConfig.NodeRole = NodeRoleWorker
	testConfig.ClusterPort = 12345
	
	configPath = filepath.Join(tempDir, "test-config.json")
	data, err := json.MarshalIndent(testConfig, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal test config: %v", err)
	}
	
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		t.Fatalf("Failed to write test config file: %v", err)
	}
	
	// Load the config file we just created
	loadedConfig, err := LoadClusterConfig(configPath)
	if err != nil {
		t.Errorf("Failed to load config file: %v", err)
	}
	
	// Verify loaded config matches what we saved
	if !loadedConfig.Enabled {
		t.Error("Loaded config should be enabled")
	}
	if loadedConfig.NodeName != "test-node" {
		t.Errorf("Expected node name 'test-node', got '%s'", loadedConfig.NodeName)
	}
func TestSaveClusterConfig(t *testing.T) {
	// Create a temp directory for config files
	tempDir, err := os.MkdirTemp("", "ollama-cluster-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create a test config to save
	testConfig := DefaultClusterConfig()
	testConfig.Enabled = true
	testConfig.NodeName = "save-test-node"
	testConfig.ClusterPort = 54321
	
	configPath := filepath.Join(tempDir, "save-test", "config.json")
	
	// Save the config
	err = SaveClusterConfig(testConfig, configPath)
	if err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}
	
	// Verify the file was created and contains valid JSON
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read saved config file: %v", err)
	}
	
	var loadedConfig ClusterConfig
	err = json.Unmarshal(data, &loadedConfig)
	if err != nil {
		t.Fatalf("Saved config is not valid JSON: %v", err)
	}
	
	// Verify the loaded config matches what we saved
	if !loadedConfig.Enabled {
		t.Error("Saved config should be enabled")
	}
	if loadedConfig.NodeName != "save-test-node" {
func TestLoadFromEnvironment(t *testing.T) {
	// Set environment variables
	os.Setenv("OLLAMA_CLUSTER_ENABLED", "true")
	os.Setenv("OLLAMA_CLUSTER_NODE_NAME", "env-test-node")
	os.Setenv("OLLAMA_CLUSTER_NODE_ROLE", "worker")
	os.Setenv("OLLAMA_CLUSTER_API_PORT", "8888")
	os.Setenv("OLLAMA_CLUSTER_HOST", "192.168.1.100")
	os.Setenv("OLLAMA_CLUSTER_PORT", "9999")
	os.Setenv("OLLAMA_CLUSTER_DISCOVERY_METHOD", "manual")
	
	// Clean up after test
	defer func() {
		os.Unsetenv("OLLAMA_CLUSTER_ENABLED")
		os.Unsetenv("OLLAMA_CLUSTER_NODE_NAME")
		os.Unsetenv("OLLAMA_CLUSTER_NODE_ROLE")
		os.Unsetenv("OLLAMA_CLUSTER_API_PORT")
		os.Unsetenv("OLLAMA_CLUSTER_HOST")
		os.Unsetenv("OLLAMA_CLUSTER_PORT")
		os.Unsetenv("OLLAMA_CLUSTER_DISCOVERY_METHOD")
	}()
	
	// Create a config and load from environment
	config := DefaultClusterConfig()
	config.LoadFromEnvironment()
	
	// Verify environment values were loaded
	if !config.Enabled {
		t.Error("Config should be enabled from env var")
	}
	if config.NodeName != "env-test-node" {
		t.Errorf("Expected node name 'env-test-node', got '%s'", config.NodeName)
	}
	if config.NodeRole != NodeRoleWorker {
		t.Errorf("Expected node role %s, got %s", NodeRoleWorker, config.NodeRole)
	}
	if config.APIPort != 8888 {
		t.Errorf("Expected API port 8888, got %d", config.APIPort)
	}
	if config.ClusterHost != "192.168.1.100" {
		t.Errorf("Expected cluster host '192.168.1.100', got '%s'", config.ClusterHost)
	}
	if config.ClusterPort != 9999 {
		t.Errorf("Expected cluster port 9999, got %d", config.ClusterPort)
	}
	if config.Discovery.Method != DiscoveryMethodManual {
		t.Errorf("Expected discovery method %s, got %s", 
			DiscoveryMethodManual, config.Discovery.Method)
	}
}

func TestGetNodeID(t *testing.T) {
	config := DefaultClusterConfig()
	
	nodeID := config.GetNodeID()
	
	// NodeID should not be empty
	if nodeID == "" {
		t.Error("Generated node ID should not be empty")
	}
	
	// NodeID should contain hostname
	hostname, _ := os.Hostname()
	if hostname != "" && !strings.Contains(nodeID, hostname) {
		t.Errorf("Expected node ID to contain hostname '%s', got '%s'", hostname, nodeID)
	}
	
	// Call again and verify it's consistent
	nodeID2 := config.GetNodeID()
	if nodeID != nodeID2 {
		t.Errorf("Node ID should be consistently generated, got '%s' and '%s'", nodeID, nodeID2)
	}
}
		t.Errorf("Expected node name 'save-test-node', got '%s'", loadedConfig.NodeName)
	}
	if loadedConfig.ClusterPort != 54321 {
		t.Errorf("Expected cluster port 54321, got %d", loadedConfig.ClusterPort)
	}
}
	if loadedConfig.NodeRole != NodeRoleWorker {
		t.Errorf("Expected node role %s, got %s", NodeRoleWorker, loadedConfig.NodeRole)
	}
	if loadedConfig.ClusterPort != 12345 {
		t.Errorf("Expected cluster port 12345, got %d", loadedConfig.ClusterPort)
	}
}
}
package cmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/spf13/cobra"
)

// Real implementation of Client for API requests
type Client struct {
	baseURL string
}

// NewClient creates a new client instance

// NewClient creates a new client instance
func NewClient() *Client {
	// Default to localhost:11434 if not specified
	baseURL := "http://localhost:11434"

	// First priority: Check if OLLAMA_HOST environment variable is set
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		baseURL = host
		// If no scheme provided, default to http://
		if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
			baseURL = "http://" + baseURL
		}
		// Ensure port is specified (default to 11434 if not)
		hostURL, err := url.Parse(baseURL)
		if err == nil && hostURL.Port() == "" {
			hostURL.Host = hostURL.Host + ":11434"
			baseURL = hostURL.String()
		}
	}

	// Second priority: Check for active cluster configuration and use first available node
	// This is critical for distributed operations to use proper node IP
	if os.Getenv("OLLAMA_CLUSTER_ENABLED") != "" || os.Getenv("OLLAMA_CLUSTER_HOST") != "" {
		// Try to load cluster configuration - do not fail if unable to
		home, err := os.UserHomeDir()
		if err == nil {
			configPath := filepath.Join(home, ".ollama", "cluster.json")
			config, err := cluster.LoadClusterConfig(configPath)
			if err == nil && config.Enabled {
				// Use the local cluster host if available
				if config.ClusterHost != "" && config.ClusterHost != "0.0.0.0" && config.ClusterHost != "localhost" {
					fmt.Printf("Using cluster host IP for API connections: %s\n", config.ClusterHost)
					baseURL = fmt.Sprintf("http://%s:%d", config.ClusterHost, config.APIPort)
				}
			}
		}
	}

	// Fix any special addresses for connectivity
	// 0.0.0.0 is a binding address, not a connection address
	if strings.Contains(baseURL, "://0.0.0.0:") {
		baseURL = strings.Replace(baseURL, "://0.0.0.0:", "://127.0.0.1:", 1)
	}

	fmt.Printf("Using API endpoint: %s\n", baseURL)
	return &Client{
		baseURL: baseURL,
	}
}

// Get sends a GET request to the Ollama API
func (c *Client) Get(path string) (*http.Response, error) {
	// Log the actual request being made
	fmt.Printf("Making GET request to %s%s\n", c.baseURL, path)
	
	// Create request
	req, err := http.NewRequest("GET", c.baseURL+path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set headers
	req.Header.Set("Content-Type", "application/json")
	
	// Execute request
	// Use same improved transport as Post method
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 0, // Disable keep-alive
		DualStack: false, // Force IPv4 only (avoid IPv6)
	}
	
	transport := &http.Transport{
		// Enable connection pooling for better stability
		DisableKeepAlives: false,
		// Set longer timeouts for better reliability
		ResponseHeaderTimeout: 5 * time.Minute,
		ExpectContinueTimeout: 2 * time.Minute,
		TLSHandshakeTimeout: 30 * time.Second,
		// Add connection management
		MaxIdleConns: 10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout: 10 * time.Minute,
		// Use IPv4-only dialer
		DialContext: dialer.DialContext,
	}
	
	client := &http.Client{
		Timeout: 15 * time.Minute, // Extended timeout for large model operations
		Transport: transport,
	}
	
	// Log the request details
	fmt.Printf("Sending GET request with timeout: %v\n", client.Timeout)
	
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	
	return resp, nil
}

// Post sends a POST request to the Ollama API
func (c *Client) Post(path, contentType string, body io.Reader) (*http.Response, error) {
	// Log the actual request being made
	fmt.Printf("Making POST request to %s%s\n", c.baseURL, path)
	
	// Create request
	req, err := http.NewRequest("POST", c.baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set headers
	req.Header.Set("Content-Type", contentType)
	
	// Execute request with improved connection handling
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 0, // Disable keep-alive
		DualStack: false, // Force IPv4 only (avoid IPv6)
	}
	
	transport := &http.Transport{
		// Enable connection pooling (changed from DisableKeepAlives: true)
		DisableKeepAlives: false,
		// Set longer timeouts for better reliability
		ResponseHeaderTimeout: 5 * time.Minute,
		ExpectContinueTimeout: 2 * time.Minute,
		TLSHandshakeTimeout: 30 * time.Second,
		// Add connection management
		MaxIdleConns: 10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout: 10 * time.Minute,
		// Use IPv4-only dialer
		DialContext: dialer.DialContext,
	}
	
	client := &http.Client{
		Timeout: 15 * time.Minute, // Extended timeout for large model operations
		Transport: transport,
	}
	
	// Log the request details
	fmt.Printf("Sending request with timeout: %v\n", client.Timeout)
	
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	
	return resp, nil
}

// Default config file path
const defaultConfigPath = "$HOME/.ollama/cluster.json"

// ClusterStartHandler starts the Ollama cluster mode
func ClusterStartHandler(cmd *cobra.Command, _ []string) error {
	// Get configuration path
	configPath, _ := cmd.Flags().GetString("config")
	if configPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to get user home directory: %w", err)
		}
		configPath = filepath.Join(home, ".ollama", "cluster.json")
	}

	// Load or create config
	config, err := cluster.LoadClusterConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load cluster configuration: %w", err)
	}

	// Override config with flags if provided
	if enabled, _ := cmd.Flags().GetBool("enabled"); enabled {
		config.Enabled = true
	}

	if nodeName, _ := cmd.Flags().GetString("node-name"); nodeName != "" {
		config.NodeName = nodeName
	}

	if nodeRole, _ := cmd.Flags().GetString("node-role"); nodeRole != "" {
		switch nodeRole {
		case "coordinator":
			config.NodeRole = cluster.NodeRoleCoordinator
		case "worker":
			config.NodeRole = cluster.NodeRoleWorker
		case "mixed":
			config.NodeRole = cluster.NodeRoleMixed
		default:
			return fmt.Errorf("invalid node role: %s (must be coordinator, worker, or mixed)", nodeRole)
		}
	}

	if apiHost, _ := cmd.Flags().GetString("api-host"); apiHost != "" {
		config.APIHost = apiHost
	}

	if apiPort, _ := cmd.Flags().GetInt("api-port"); apiPort != 0 {
		config.APIPort = apiPort
	}

	if clusterHost, _ := cmd.Flags().GetString("cluster-host"); clusterHost != "" {
		config.ClusterHost = clusterHost
	}

	if clusterPort, _ := cmd.Flags().GetInt("cluster-port"); clusterPort != 0 {
		config.ClusterPort = clusterPort
	}

	// Load environment variables
	config.LoadFromEnvironment()

	// No need to save the config to disk - in-memory approach
	fmt.Printf("Using in-memory cluster configuration\n")

	// Start cluster mode if enabled
	if !config.Enabled {
		return fmt.Errorf("cluster mode is not enabled in configuration")
	}

	// Create and initialize cluster mode
	clusterMode, err := cluster.NewClusterMode(config)
	if err != nil {
		return fmt.Errorf("failed to initialize cluster mode: %w", err)
	}

	// Start cluster services
	if err := clusterMode.Start(); err != nil {
		return fmt.Errorf("failed to start cluster mode: %w", err)
	}

	fmt.Println("Cluster mode started successfully")
	return nil
}

// ClusterStatusHandler shows the status of the cluster
func ClusterStatusHandler(cmd *cobra.Command, _ []string) error {
	configPath, _ := cmd.Flags().GetString("config")
	if configPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to get user home directory: %w", err)
		}
		configPath = filepath.Join(home, ".ollama", "cluster.json")
	}

	// Load config
	config, err := cluster.LoadClusterConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load cluster configuration: %w", err)
	}

	if !config.Enabled {
		fmt.Println("Cluster mode is not enabled")
		return nil
	}

	// In a real implementation, we would query the actual cluster status
	// For now, just show the configuration
	fmt.Println("Cluster Configuration:")
	fmt.Printf("  Node Name: %s\n", config.NodeName)
	fmt.Printf("  Node Role: %s\n", config.NodeRole)
	fmt.Printf("  API: %s:%d\n", config.APIHost, config.APIPort)
	fmt.Printf("  Cluster: %s:%d\n", config.ClusterHost, config.ClusterPort)
	fmt.Printf("  Discovery Method: %s\n", config.Discovery.Method)

	if config.Discovery.Method == cluster.DiscoveryMethodMulticast {
		fmt.Printf("  Multicast Address: %s\n", config.Discovery.MulticastAddress)
	} else if config.Discovery.Method == cluster.DiscoveryMethodManual {
		fmt.Printf("  Node List: %v\n", config.Discovery.NodeList)
	}

	return nil
}

// REMOVED: duplicate function declaration
// ClusterJoinHandler joins the node to an existing cluster
func ClusterJoinHandler(cmd *cobra.Command, _ []string) error {
	// Get the node to join
	nodeHost, _ := cmd.Flags().GetString("node-host")
	if nodeHost == "" {
		return fmt.Errorf("node-host is required")
	}

	// Get the node port
	nodePort, _ := cmd.Flags().GetInt("node-port")
	if nodePort <= 0 {
		return fmt.Errorf("node-port is required and must be > 0")
	}

	// Get optional parameters
	nodeRole, _ := cmd.Flags().GetString("node-role")
	joinToken, _ := cmd.Flags().GetString("join-token")
	forceReplace, _ := cmd.Flags().GetBool("force-replace")

	// Prepare request body
	reqBody := map[string]interface{}{
		"node_host":     nodeHost,
		"node_port":     nodePort,
		"join_token":    joinToken,
		"node_role":     nodeRole,
		"force_replace": forceReplace,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	// Make API call to join the cluster
	client := NewClient()
	resp, err := client.Post("/api/cluster/join", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to join cluster: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Check for success
	if success, ok := result["success"].(bool); !ok || !success {
		errMsg := "unknown error"
		if msg, ok := result["error"].(string); ok {
			errMsg = msg
		}
		return fmt.Errorf("failed to join cluster: %s", errMsg)
	}

	// Display success message
	fmt.Println("Successfully joined cluster:")
	if nodeID, ok := result["node_id"].(string); ok {
		fmt.Printf("  Node ID: %s\n", nodeID)
	}
	if clusterID, ok := result["cluster_id"].(string); ok {
		fmt.Printf("  Cluster ID: %s\n", clusterID)
	}

	return nil
}

// ClusterLeaveHandler removes the node from its current cluster
func ClusterLeaveHandler(cmd *cobra.Command, _ []string) error {
	// Get optional parameters
	graceful, _ := cmd.Flags().GetBool("graceful")
	timeout, _ := cmd.Flags().GetInt("timeout")
	nodeID, _ := cmd.Flags().GetString("node-id")

	// Prepare request body
	reqBody := map[string]interface{}{
		"graceful":        graceful,
		"timeout_seconds": timeout,
	}
	
	if nodeID != "" {
		reqBody["node_id"] = nodeID
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	// Make API call to leave the cluster
	client := NewClient()
	resp, err := client.Post("/api/cluster/leave", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to leave cluster: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Check for success
	if success, ok := result["success"].(bool); !ok || !success {
		errMsg := "unknown error"
		if msg, ok := result["error"].(string); ok {
			errMsg = msg
		}
		return fmt.Errorf("failed to leave cluster: %s", errMsg)
	}

	fmt.Println("Successfully left the cluster")
	return nil
}

// ClusterNodesHandler lists all nodes in the cluster
func ClusterNodesHandler(cmd *cobra.Command, _ []string) error {
	// Make API call to get nodes
	client := NewClient()
	fmt.Println("Fetching cluster nodes from Ollama server...")
	
	resp, err := client.Get("/api/cluster/nodes")
	if err != nil {
		return fmt.Errorf("failed to fetch cluster nodes: %w", err)
	}
	defer resp.Body.Close()
	
	// Read the response body for debugging
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}
	
	// Create a new reader with the same bytes for JSON decoding
	resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	
	// Log response for debugging
	fmt.Printf("Response status code: %d\n", resp.StatusCode)
	fmt.Printf("Response body: %s\n", string(bodyBytes))
	
	// Try to parse response in different formats
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		fmt.Printf("Warning: Failed to parse response as JSON object: %v\n", err)
		
		// Try parsing as array directly
		resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		var arrayResult []interface{}
		if err := json.NewDecoder(resp.Body).Decode(&arrayResult); err != nil {
			return fmt.Errorf("failed to parse response in any format: %w", err)
		}
		
		// Handle array response
		fmt.Println("Response parsed as array format")
		nodes := arrayResult
		fmt.Println("Cluster Nodes:")
		displayNodes(nodes)
		return nil
	}
	
	// Check for error in response
	if errMsg, ok := result["error"].(string); ok {
		return fmt.Errorf("API error: %s", errMsg)
	}
	
	// Extract nodes with more robust handling
	var nodes []interface{}
	var ok bool
	
	// Try different possible formats the server might return
	if nodes, ok = result["nodes"].([]interface{}); !ok {
		// If the response itself is an array, try using that
		var arrayResult []interface{}
		if err := json.Unmarshal(bodyBytes, &arrayResult); err == nil && len(arrayResult) > 0 {
			nodes = arrayResult
			fmt.Println("Using array response format")
		} else if len(result) > 0 {
			// If no "nodes" field but we have other fields, the response itself might be a single node
			// Convert the single node to an array for consistent handling
			fmt.Println("Response might be a single node, converting to array")
			nodes = []interface{}{result}
		} else {
			return fmt.Errorf("unexpected response format: no 'nodes' field found in response")
		}
	}
	
	// Display nodes
	fmt.Println("Cluster Nodes:")
	displayNodes(nodes)
	
	return nil
}

// displayNodes handles the display of node information consistently
func displayNodes(nodes []interface{}) {
	if len(nodes) == 0 {
		fmt.Println("  No nodes found in cluster")
		return
	}

	for _, n := range nodes {
		node, ok := n.(map[string]interface{})
		if !ok {
			fmt.Printf("  Warning: Received unexpected node format: %T\n", n)
			continue
		}

		// Extract fields with type safety
		id, _ := node["id"].(string)
		name, _ := node["name"].(string)
		role, _ := node["role"].(string)
		status, _ := node["status"].(string)
		address, _ := node["address"].(string)

		// If id is empty, try alternatives
		if id == "" {
			if tempID, ok := node["node_id"].(string); ok {
				id = tempID
			} else {
				id = "unknown"
			}
		}

		// If name is empty, use a default
		if name == "" {
			name = "unnamed-node"
		}

		fmt.Printf("  %s (%s)\n", name, id)
		if role != "" {
			fmt.Printf("    Role: %s\n", role)
		}
		if status != "" {
			fmt.Printf("    Status: %s\n", status)
		}
		if address != "" {
			fmt.Printf("    Address: %s\n", address)
		}
		
		// Display models if available
		if models, ok := node["models"].([]interface{}); ok && len(models) > 0 {
			fmt.Printf("    Models: ")
			for i, m := range models {
				if i > 0 {
					fmt.Printf(", ")
				}
				fmt.Printf("%s", m)
			}
			fmt.Println()
		}
		
		fmt.Println()
	}
}

// ClusterModelLoadHandler loads a model in cluster mode
func ClusterModelLoadHandler(cmd *cobra.Command, _ []string) error {
	// Get required parameters
	model, _ := cmd.Flags().GetString("model")
	if model == "" {
		return fmt.Errorf("model name is required")
	}

	// Get optional parameters
	distributed, _ := cmd.Flags().GetBool("distributed")
	shardCount, _ := cmd.Flags().GetInt("shards")
	strategy, _ := cmd.Flags().GetString("strategy")
	nodeIDs, _ := cmd.Flags().GetStringSlice("node-ids")

	// Prepare request body
	reqBody := map[string]interface{}{
		"model":       model,
		"distributed": distributed,
	}
	
	if shardCount > 0 {
		reqBody["shard_count"] = shardCount
	}
	
	if strategy != "" {
		reqBody["strategy"] = strategy
	}
	
	if len(nodeIDs) > 0 {
		reqBody["node_ids"] = nodeIDs
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}

	// Make API call to load the model
	client := NewClient()
	
	// Before making the request, perform client-side connection diagnostics
	fmt.Println("Performing client-side connection diagnostics...")
	
	// Test the server connection before making the actual request
	testURL := fmt.Sprintf("%s/api/version", client.baseURL)
	fmt.Printf("Testing server connection to: %s\n", testURL)
	
	// Use the same IPv4-only transport config we'll use for the actual request
	testDialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 0,
		DualStack: false, // Force IPv4 only
	}
	
	transport := &http.Transport{
		DisableKeepAlives: false, // Enable keep-alives for better connection stability
		ResponseHeaderTimeout: 2 * time.Minute,
		TLSHandshakeTimeout: 30 * time.Second,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout: 10 * time.Minute,
		DialContext: testDialer.DialContext,
	}
	
	testClient := &http.Client{Timeout: 10 * time.Second, Transport: transport}
	
	testResp, testErr := testClient.Get(testURL)
	if testErr != nil {
		fmt.Printf("⚠️ SERVER CONNECTION TEST FAILED: %v\n", testErr)
	} else {
		defer testResp.Body.Close()
		fmt.Printf("✓ Server connection test passed (status: %d)\n", testResp.StatusCode)
	}
	
	// Perform the actual model loading request
	fmt.Println("Sending model load request...")
	// Create detailed logging to trace the entire request flow
	fmt.Println("==================== BEGIN DETAILED DIAGNOSTICS ====================")
	fmt.Printf("Request details:\n- URL: %s/api/cluster/model/load\n- Content-Type: application/json\n- Body length: %d bytes\n",
		client.baseURL, len(jsonData))
	
	// Create request with context and detailed instrumentation
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, "POST", client.baseURL+"/api/cluster/model/load", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("Error creating request: %T - %v\n", err, err)
		return fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set proper headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("User-Agent", "Ollama-Client/Diagnostics")
	req.Header.Set("X-Debug-Mode", "true")
	
	// Log all request headers
	fmt.Println("Request headers:")
	for key, values := range req.Header {
		fmt.Printf("- %s: %v\n", key, values)
	}
	
	// Custom transport with extra logging
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
		DualStack: true,
	}
	
	transport = &http.Transport{
		DisableKeepAlives: false,
		// Extended timeouts
		ResponseHeaderTimeout: 5 * time.Minute,
		ExpectContinueTimeout: 2 * time.Minute,
		TLSHandshakeTimeout: 30 * time.Second,
		// Add connection management
		MaxIdleConns: 10,
		MaxIdleConnsPerHost: 5,
		IdleConnTimeout: 10 * time.Minute,
		// Use instrumented dialer
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			fmt.Printf("Dialing connection to %s (%s)...\n", addr, network)
			conn, err := dialer.DialContext(ctx, network, addr)
			if err != nil {
				fmt.Printf("Connection error: %T - %v\n", err, err)
				return nil, err
			}
			fmt.Printf("Connection established to %s (local=%s, remote=%s)\n",
				addr, conn.LocalAddr().String(), conn.RemoteAddr().String())
			
			// Wrap connection with logging
			return &loggingConn{Conn: conn, addr: addr}, nil
		},
	}
	
	client2 := &http.Client{
		Timeout: 15 * time.Minute,
		Transport: transport,
	}
	
	// Execute the request with detailed logging
	fmt.Println("Sending request...")
	startTime := time.Now()
	resp, err := client2.Do(req)
	elapsedTime := time.Since(startTime)
	
	if err != nil {
		fmt.Printf("Error after %v: %T - %v\n", elapsedTime, err, err)
		
		// Detailed error analysis
		var netErr net.Error
		if errors.As(err, &netErr) {
			fmt.Printf("Network error detected:\n- Timeout: %v\n- Temporary: %v\n",
				netErr.Timeout(), netErr.Temporary())
		}
		
		// Check for specific error messages
		errStr := err.Error()
		if strings.Contains(errStr, "forcibly closed") {
			fmt.Println("Connection forcibly closed by remote host - this may indicate:")
			fmt.Println("1. Server-side timeout or resource constraint")
			fmt.Println("2. Network/firewall interruption")
			fmt.Println("3. Server process crash or restart")
		}
		
		fmt.Println("==================== END DETAILED DIAGNOSTICS ====================")
		return fmt.Errorf("failed to load model in cluster: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Check for success
	if success, ok := result["success"].(bool); !ok || !success {
		errMsg := "unknown error"
		if msg, ok := result["error"].(string); ok {
			errMsg = msg
		}
		return fmt.Errorf("failed to load model in cluster: %s", errMsg)
	}

	// Display success message
	fmt.Println("Successfully loaded model in cluster mode:")
	fmt.Printf("  Model: %s\n", model)
	
	// Use the distributed status from the response, not the request
	responseDistributed := distributed
	if respDistributed, ok := result["distributed"].(bool); ok {
		responseDistributed = respDistributed
	}
	fmt.Printf("  Distributed: %v\n", responseDistributed)
	
	// Show which nodes the model was loaded on
	if nodes, ok := result["nodes"].([]interface{}); ok && len(nodes) > 0 {
		fmt.Printf("  Loaded on nodes: ")
		for i, n := range nodes {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%s", n.(string))
		}
		fmt.Println()
	}

	return nil
}

// ClusterRunHandler runs a model in the cluster
func ClusterRunHandler(cmd *cobra.Command, args []string) error {
	// Get required parameters
	model, _ := cmd.Flags().GetString("model")
	if model == "" {
		return fmt.Errorf("model name is required")
	}
	
	// Get optional parameters
	systemPrompt, _ := cmd.Flags().GetString("system")
	temperature, _ := cmd.Flags().GetFloat32("temperature")
	format, _ := cmd.Flags().GetString("format")
	
	// Prepare request body
	reqBody := map[string]interface{}{
		"model": model,
		"stream": true,
	}
	
	if temperature != 0 {
		reqBody["temperature"] = temperature
	}
	
	if format != "" {
		reqBody["format"] = format
	}
	
	// If args were provided, use them as the prompt
	if len(args) > 0 {
		prompt := strings.Join(args, " ")
		reqBody["prompt"] = prompt
	} else {
		// No prompt provided, check for system prompt only
		if systemPrompt == "" {
			return fmt.Errorf("either prompt text or --system prompt is required")
		}
	}
	
	// Add system prompt if provided
	if systemPrompt != "" {
		reqBody["system"] = systemPrompt
	}
	
	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to encode request: %w", err)
	}
	
	// Debug: Print the request body
	fmt.Printf("Running model '%s' in cluster mode...\n", model)
	fmt.Printf("Request body: %s\n", string(jsonData))
	
	// Make API call to run the model
	client := NewClient()
	resp, err := client.Post("/api/cluster/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to run model in cluster: %w", err)
	}
	defer resp.Body.Close()
	
	// For streaming responses, we need to read line by line
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		// Parse each line as JSON
		var streamResp map[string]interface{}
		if err := json.Unmarshal([]byte(line), &streamResp); err != nil {
			fmt.Printf("Error parsing response: %v\n", err)
			continue
		}
		
		// Check for error
		if errMsg, ok := streamResp["error"].(string); ok {
			return fmt.Errorf("API error: %s", errMsg)
		}
		
		// Print generated text
		if response, ok := streamResp["response"].(string); ok {
			fmt.Print(response)
		}
		
		// Check for done flag
		if done, ok := streamResp["done"].(bool); ok && done {
			fmt.Println() // Add a newline at the end
			break
		}
	}
	
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading response: %w", err)
	}
	
	return nil
}

// Update the RegisterClusterCommands function to add the new commands
func RegisterClusterCommands(root *cobra.Command) {
	clusterCmd := &cobra.Command{
		Use:   "cluster",
		Short: "Cluster mode commands",
	}

	startCmd := &cobra.Command{
		Use:   "start",
		Short: "Start cluster mode",
		RunE:  ClusterStartHandler,
	}

	statusCmd := &cobra.Command{
		Use:   "status",
		Short: "Show cluster status",
		RunE:  ClusterStatusHandler,
	}
	
	joinCmd := &cobra.Command{
		Use:   "join",
		Short: "Join an existing cluster",
		RunE:  ClusterJoinHandler,
	}
	
	leaveCmd := &cobra.Command{
		Use:   "leave",
		Short: "Leave the current cluster",
		RunE:  ClusterLeaveHandler,
	}
	
	nodesCmd := &cobra.Command{
		Use:   "nodes",
		Short: "List all nodes in the cluster",
		RunE:  ClusterNodesHandler,
	}
	
	modelLoadCmd := &cobra.Command{
		Use:   "model",
		Short: "Load a model in cluster mode",
		RunE:  ClusterModelLoadHandler,
	}
	
	runCmd := &cobra.Command{
		Use:   "run [prompt]",
		Short: "Run a model in cluster mode directly from terminal",
		Args:  cobra.ArbitraryArgs,
		RunE:  ClusterRunHandler,
	}

	// Configure flags for the start command
	startCmd.Flags().String("config", "", "Path to cluster configuration file")
	startCmd.Flags().Bool("enabled", false, "Enable cluster mode")
	startCmd.Flags().String("node-name", "", "Name of this node in the cluster")
	startCmd.Flags().String("node-role", "", "Role of this node (coordinator, worker, or mixed)")
	startCmd.Flags().String("api-host", "", "API server host address")
	startCmd.Flags().Int("api-port", 0, "API server port")
	startCmd.Flags().String("cluster-host", "", "Cluster communication host address")
	startCmd.Flags().Int("cluster-port", 0, "Cluster communication port")
	startCmd.Flags().String("discovery", "", "Discovery method (multicast or manual)")
	startCmd.Flags().String("multicast-address", "", "Multicast address for discovery")
	startCmd.Flags().StringSlice("node-list", []string{}, "List of known nodes for manual discovery")

	// Configure flags for the status command
	statusCmd.Flags().String("config", "", "Path to cluster configuration file")
	
	// Configure flags for the join command
	joinCmd.Flags().String("node-host", "", "Host address of the node to join")
	joinCmd.Flags().Int("node-port", 0, "Port of the node to join")
	joinCmd.Flags().String("node-role", "", "Role to assume in the cluster (coordinator, worker, or mixed)")
	joinCmd.Flags().String("join-token", "", "Authentication token for joining the cluster")
	joinCmd.Flags().Bool("force-replace", false, "Force replace existing node with same ID")
	joinCmd.MarkFlagRequired("node-host")
	joinCmd.MarkFlagRequired("node-port")
	
	// Configure flags for the leave command
	leaveCmd.Flags().Bool("graceful", true, "Perform a graceful exit, migrating workloads if possible")
	leaveCmd.Flags().Int("timeout", 30, "Timeout in seconds for graceful exit")
	leaveCmd.Flags().String("node-id", "", "ID of the node to remove (if not this node)")
	
	// Configure flags for the model load command
	modelLoadCmd.Flags().String("model", "", "Model to load")
	modelLoadCmd.Flags().Bool("distributed", false, "Load model in distributed mode")
	modelLoadCmd.Flags().Int("shards", 0, "Number of shards for distributed loading")
	modelLoadCmd.Flags().String("strategy", "auto", "Distribution strategy (auto, memory-optimized, speed-optimized)")
	modelLoadCmd.Flags().StringSlice("node-ids", []string{}, "Specific nodes to load the model on")
	modelLoadCmd.MarkFlagRequired("model")
	
	// Configure flags for the run command
	runCmd.Flags().String("model", "", "Model to run")
	runCmd.Flags().String("system", "", "System prompt to use")
	runCmd.Flags().Float32("temperature", 0.7, "Temperature for sampling (0.0 to 1.0)")
	runCmd.Flags().String("format", "json", "Response format (json or text)")
	runCmd.MarkFlagRequired("model")

	clusterCmd.AddCommand(startCmd, statusCmd, joinCmd, leaveCmd, nodesCmd, modelLoadCmd, runCmd)
	root.AddCommand(clusterCmd)
}

// loggingConn wraps a net.Conn with logging
type loggingConn struct {
	net.Conn
	addr string
}

func (c *loggingConn) Read(b []byte) (n int, err error) {
	n, err = c.Conn.Read(b)
	if err != nil && err != io.EOF {
		fmt.Printf("READ ERROR to %s: %v (read %d bytes)\n", c.addr, err, n)
	}
	return
}

func (c *loggingConn) Write(b []byte) (n int, err error) {
	n, err = c.Conn.Write(b)
	if err != nil {
		fmt.Printf("WRITE ERROR to %s: %v (wrote %d bytes)\n", c.addr, err, n)
	}
	return
}

func (c *loggingConn) Close() error {
	fmt.Printf("Connection to %s being closed\n", c.addr)
	return c.Conn.Close()
}
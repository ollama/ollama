package api

import (
	"fmt"
	"net"
	"os"
	"reflect"
	"time"
)

// ClusterNodeResponse represents information about a node in the cluster
type ClusterNodeResponse struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Role        string   `json:"role"`
	Status      string   `json:"status"`
	Address     string   `json:"address"`
	JoinedAt    time.Time `json:"joined_at"`
	LastContact time.Time `json:"last_contact,omitempty"`
	Models      []string `json:"models,omitempty"`
	Resources   *NodeResources `json:"resources,omitempty"`
}

// NodeResources represents the resource information for a node
type NodeResources struct {
	CPUCores    int     `json:"cpu_cores"`
	CPUUsage    float64 `json:"cpu_usage"`
	TotalMemory int64   `json:"total_memory"`
	UsedMemory  int64   `json:"used_memory"`
	GPUCount    int     `json:"gpu_count"`
	GPUModels   []string `json:"gpu_models,omitempty"`
	GPUMemory   []int64  `json:"gpu_memory,omitempty"`
}

// ClusterStatusResponse represents the overall status of the cluster
type ClusterStatusResponse struct {
	Enabled     bool                 `json:"enabled"`
	Mode        string               `json:"mode"`
	NodeCount   int                  `json:"node_count"`
	CurrentNode ClusterNodeResponse  `json:"current_node"`
	Nodes       []ClusterNodeResponse `json:"nodes"`
	Models      []ClusterModelResponse `json:"models"`
	StartedAt   time.Time            `json:"started_at"`
	Healthy     bool                 `json:"healthy"`
}

// ClusterModelResponse represents information about a model distributed in the cluster
type ClusterModelResponse struct {
	Name        string    `json:"name"`
	Size        int64     `json:"size"`
	Distributed bool      `json:"distributed"`
	Nodes       []string  `json:"nodes"`
	Shards      int       `json:"shards"`
	Status      string    `json:"status"`
	LoadedAt    time.Time `json:"loaded_at"`
}

// ClusterJoinRequest represents a request to join an existing cluster
type ClusterJoinRequest struct {
	NodeHost     string `json:"node_host"`
	NodePort     int    `json:"node_port"`
	NodeRole     string `json:"node_role,omitempty"`
	JoinToken    string `json:"join_token,omitempty"`
	ForceReplace bool   `json:"force_replace,omitempty"`
}

// ClusterJoinResponse represents the response to a join request
type ClusterJoinResponse struct {
	Success     bool   `json:"success"`
	NodeID      string `json:"node_id,omitempty"`
	ClusterID   string `json:"cluster_id,omitempty"`
	NodesJoined int    `json:"nodes_joined,omitempty"`
	Error       string `json:"error,omitempty"`
}

// ClusterLeaveRequest represents a request to leave the cluster
type ClusterLeaveRequest struct {
	NodeID   string `json:"node_id,omitempty"`
	Graceful bool   `json:"graceful,omitempty"`
	Timeout  int    `json:"timeout_seconds,omitempty"`
}

// ClusterLeaveResponse represents the response to a leave request
type ClusterLeaveResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// ClusterModelLoadRequest represents a request to load a model in cluster mode
type ClusterModelLoadRequest struct {
	Model       string   `json:"model"`
	Distributed bool     `json:"distributed,omitempty"`
	ShardCount  int      `json:"shard_count,omitempty"`
	Strategy    string   `json:"strategy,omitempty"`
	NodeIDs     []string `json:"node_ids,omitempty"`
}

// ClusterModelLoadResponse represents the response to a model load request
type ClusterModelLoadResponse struct {
	Success     bool     `json:"success"`
	Model       string   `json:"model"`
	Distributed bool     `json:"distributed"`
	Nodes       []string `json:"nodes,omitempty"`
	Error       string   `json:"error,omitempty"`
}

// ConvertNodeInfoToResponse converts a NodeInfo struct from the cluster package to an API response
func ConvertNodeInfoToResponse(nodeInfo interface{}) ClusterNodeResponse {
	// Create empty response to populate
	response := ClusterNodeResponse{}
	
	// Try direct type assertion as a cluster.NodeInfo struct with proper fields
	type nodeInfoStruct struct {
		ID           string
		Name         string
		Role         string
		Status       string
		Addr         net.IP
		ApiPort      int
		ClusterPort  int
		LastHeartbeat time.Time
		Resources    struct {
			CPUCores    int
			MemoryMB    uint64
			GPUCount    int
			GPUModels   []string
			GPUMemoryMB []uint64
		}
		Version string
	}
	
	// Use reflection to safely extract values from the nodeInfoStruct
	val := reflect.ValueOf(nodeInfo)
	
	// Check if we got a pointer and dereference it if needed
	if val.Kind() == reflect.Ptr && !val.IsNil() {
		val = val.Elem()
	}
	
	// Only process if we have a struct
	if val.Kind() == reflect.Struct {
		// Extract ID field
		idField := val.FieldByName("ID")
		if idField.IsValid() && idField.Kind() == reflect.String {
			response.ID = idField.String()
		}
		
		// Extract Name field
		nameField := val.FieldByName("Name")
		if nameField.IsValid() && nameField.Kind() == reflect.String {
			response.Name = nameField.String()
		}
		
		// Extract Role field
		roleField := val.FieldByName("Role")
		if roleField.IsValid() {
			response.Role = fmt.Sprintf("%v", roleField.Interface())
		}
		
		// Extract Status field
		statusField := val.FieldByName("Status")
		if statusField.IsValid() {
			response.Status = fmt.Sprintf("%v", statusField.Interface())
		}
		
		// Extract and format address
		addrField := val.FieldByName("Addr")
		apiPortField := val.FieldByName("ApiPort")
		
		if addrField.IsValid() && apiPortField.IsValid() {
			// Convert net.IP to string safely
			var addrStr string
			if ip, ok := addrField.Interface().(net.IP); ok && ip != nil {
				addrStr = ip.String()
			} else if addrField.Kind() == reflect.String {
				addrStr = addrField.String()
			}
			
			// Get API port as int
			var apiPort int
			if apiPortField.Kind() == reflect.Int {
				apiPort = int(apiPortField.Int())
			}
			
			// Combine into address:port format
			if addrStr != "" && apiPort > 0 {
				response.Address = fmt.Sprintf("%s:%d", addrStr, apiPort)
			} else if addrStr != "" {
				response.Address = addrStr
			}
		}
		
		// Extract LastHeartbeat for timestamps
		lastHeartbeatField := val.FieldByName("LastHeartbeat")
		if lastHeartbeatField.IsValid() && lastHeartbeatField.Type().String() == "time.Time" {
			if t, ok := lastHeartbeatField.Interface().(time.Time); ok {
				response.JoinedAt = t
				response.LastContact = t
			}
		}
		
		// Extract Resources if available
		resourcesField := val.FieldByName("Resources")
		if resourcesField.IsValid() && resourcesField.Kind() == reflect.Struct {
			// Create resources object
			resources := &NodeResources{}
			
			// CPU cores
			cpuCoresField := resourcesField.FieldByName("CPUCores")
			if cpuCoresField.IsValid() && cpuCoresField.Kind() == reflect.Int {
				resources.CPUCores = int(cpuCoresField.Int())
			}
			
			// Memory
			memoryMBField := resourcesField.FieldByName("MemoryMB")
			if memoryMBField.IsValid() {
				resources.TotalMemory = int64(memoryMBField.Uint()) * 1024 * 1024 // Convert MB to bytes
			}
			
			// GPU count and models
			gpuCountField := resourcesField.FieldByName("GPUCount")
			if gpuCountField.IsValid() && gpuCountField.Kind() == reflect.Int {
				resources.GPUCount = int(gpuCountField.Int())
			}
			
			gpuModelsField := resourcesField.FieldByName("GPUModels")
			if gpuModelsField.IsValid() && gpuModelsField.Kind() == reflect.Slice {
				resources.GPUModels = make([]string, gpuModelsField.Len())
				for i := 0; i < gpuModelsField.Len(); i++ {
					if model, ok := gpuModelsField.Index(i).Interface().(string); ok {
						resources.GPUModels[i] = model
					}
				}
			}
			
			// Set resources to response
			response.Resources = resources
		}
	} else {
		// Fallback method - attempt to get fields using type assertions
		type nodeInfoMap map[string]interface{}
		if m, ok := nodeInfo.(nodeInfoMap); ok {
			// Extract basic fields from map
			if id, ok := m["ID"].(string); ok {
				response.ID = id
			}
			if name, ok := m["Name"].(string); ok {
				response.Name = name
			}
			if role, ok := m["Role"].(string); ok {
				response.Role = role
			} else if role, ok := m["Role"].(fmt.Stringer); ok {
				response.Role = role.String()
			}
			if status, ok := m["Status"].(string); ok {
				response.Status = status
			} else if status, ok := m["Status"].(fmt.Stringer); ok {
				response.Status = status.String()
			}
			
			// Extract address information
			var addrStr string
			
			if addr, ok := m["Addr"].(net.IP); ok && addr != nil {
				addrStr = addr.String()
			} else if addr, ok := m["Addr"].(string); ok {
				addrStr = addr
			}
			
			if apiPort, ok := m["ApiPort"].(int); ok && apiPort > 0 {
				response.Address = fmt.Sprintf("%s:%d", addrStr, apiPort)
			} else {
				response.Address = addrStr
			}
		}
	}
	
	// If we got an empty response, generate a meaningful unique ID
	if response.ID == "" {
		// Try to get hostname for better identification
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "unknown-host"
		}
		
		// Create a unique ID based on hostname and current time
		uniqueId := fmt.Sprintf("%s-%d", hostname, time.Now().UnixNano())
		
		response.ID = uniqueId
		response.Name = hostname
		response.Role = "mixed"  // Default role
		response.Status = "online"  // Default status
		
		// Try to get a local IP address for the node
		addrs, err := net.InterfaceAddrs()
		if err == nil {
			for _, addr := range addrs {
				if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
					response.Address = fmt.Sprintf("%s:11434", ipnet.IP.String())
					break
				}
			}
		}
		
		if response.Address == "" {
			response.Address = "localhost:11434"  // Default address if nothing else works
		}
		
		response.JoinedAt = time.Now()
		response.LastContact = time.Now()
	}
	
	return response
}
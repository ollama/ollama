package model

import (
	"fmt"
	"sync"
)

// RoutingTable manages the communication paths between model partitions
type RoutingTable struct {
	// partitioner provides access to model partitioning information
	partitioner *ModelPartitioner
	
	// routes maps source node IDs to their destination node IDs for each model
	// modelID → sourceNodeID → []destinationNodeID
	routes map[string]map[string][]string
	
	// routeParams stores additional routing parameters per connection
	routeParams map[string]map[string]map[string]RouteParameters
	
	// mu protects the routes map
	mu sync.RWMutex
}

// RouteParameters contains configuration for a specific inter-node route
type RouteParameters struct {
	// Priority defines the importance of this route (higher = more important)
	Priority int
	
	// CompressionLevel controls tensor compression on this route
	CompressionLevel int
	
	// Bandwidth estimates the available bandwidth in Mbps
	Bandwidth int
	
	// Latency estimates the typical latency in milliseconds
	Latency int
}

// RouteType classifies the kind of communication needed
type RouteType string

const (
	// RouteTypeForward for sending activations to the next partition
	RouteTypeForward RouteType = "forward"
	
	// RouteTypeBackward for sending gradients back during training
	RouteTypeBackward RouteType = "backward"
	
	// RouteTypeSync for synchronizing model updates
	RouteTypeSync RouteType = "sync"
	
	// RouteTypeAll for all communication types
	RouteTypeAll RouteType = "all"
)

// RoutingError represents errors that can occur during routing
type RoutingError struct {
	Err error
	SourceNodeID string
	DestNodeID   string
}

func (e RoutingError) Error() string {
	return fmt.Sprintf("routing error from node %s to %s: %v", e.SourceNodeID, e.DestNodeID, e.Err)
}
// NewRoutingTable creates a new routing table for model partitions
func NewRoutingTable(partitioner *ModelPartitioner) *RoutingTable {
	return &RoutingTable{
		partitioner: partitioner,
		routes:      make(map[string]map[string][]string),
		routeParams: make(map[string]map[string]map[string]RouteParameters),
	}
}

// BuildRoutes constructs the routing table for a specific model
func (rt *RoutingTable) BuildRoutes(modelID string) error {
	// Get model partitioning
	partitions, exists := rt.partitioner.GetModelPartitions(modelID)
	if !exists {
		return fmt.Errorf("no partitioning found for model %s", modelID)
	}
	
	rt.mu.Lock()
	defer rt.mu.Unlock()
	
	// Initialize routes map for this model
	rt.routes[modelID] = make(map[string][]string)
	rt.routeParams[modelID] = make(map[string]map[string]RouteParameters)
	
	// Create routing based on partition dependencies
	// For layer-wise partitioning, each node needs to communicate with 
	// the nodes handling adjacent layers
	for i, partition := range partitions {
		sourceNodeID := partition.NodeID
		
		// Initialize params map for this source node
		if _, exists := rt.routeParams[modelID][sourceNodeID]; !exists {
			rt.routeParams[modelID][sourceNodeID] = make(map[string]RouteParameters)
		}
		
		// Connect to the next partition (if not last)
		if i < len(partitions)-1 {
			nextNodeID := partitions[i+1].NodeID
			// Add forward route
			rt.addRoute(modelID, sourceNodeID, nextNodeID)
			
			// Set route parameters
			rt.routeParams[modelID][sourceNodeID][nextNodeID] = RouteParameters{
				Priority:         10,
				CompressionLevel: 5, // Medium compression
				Bandwidth:        1000,
				Latency:          5,
			}
		}
		
		// Connect to the previous partition (if not first)
		if i > 0 {
			prevNodeID := partitions[i-1].NodeID
			// Add backward route
			rt.addRoute(modelID, sourceNodeID, prevNodeID)
			
			// Set route parameters
			rt.routeParams[modelID][sourceNodeID][prevNodeID] = RouteParameters{
				Priority:         8,
				CompressionLevel: 5, // Medium compression
				Bandwidth:        1000,
				Latency:          5,
			}
		}
		
		fmt.Printf("Model %s: Established routes for node %s (partition %s)\n",
			modelID, sourceNodeID, partition.PartitionID)
	}
	
	// For tensor-wise partitioning, we also need all-to-all communication for tensor synchronization
	// In a real implementation, this would be more selective based on tensor dependencies
	for i, sourcePartition := range partitions {
		for j, destPartition := range partitions {
			// Skip self-routes
			if i == j {
				continue
			}
			
			sourceNodeID := sourcePartition.NodeID
			destNodeID := destPartition.NodeID
			
			// Add sync route
			rt.addRoute(modelID, sourceNodeID, destNodeID)
			
			// Set route parameters - sync has lower priority than direct dependencies
			rt.routeParams[modelID][sourceNodeID][destNodeID] = RouteParameters{
				Priority:         5,
				CompressionLevel: 8, // Higher compression for all-to-all communication
				Bandwidth:        500, // Assume less bandwidth availability for sync
				Latency:          10,
			}
		}
	}
	
	return nil
}

// addRoute adds a route from source to destination
func (rt *RoutingTable) addRoute(modelID, sourceNodeID, destNodeID string) {
	// Initialize if needed
	if rt.routes[modelID][sourceNodeID] == nil {
		rt.routes[modelID][sourceNodeID] = make([]string, 0)
	}
	
	// Check if route already exists
	for _, existingDest := range rt.routes[modelID][sourceNodeID] {
		if existingDest == destNodeID {
			return // Route already exists
		}
	}
	
	// Add the new route
	rt.routes[modelID][sourceNodeID] = append(rt.routes[modelID][sourceNodeID], destNodeID)
}

// GetRoutesForNode returns all routes from a specific node for a model
func (rt *RoutingTable) GetRoutesForNode(modelID, nodeID string) ([]string, error) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	
	if modelRoutes, ok := rt.routes[modelID]; ok {
		if nodeRoutes, ok := modelRoutes[nodeID]; ok {
			return nodeRoutes, nil
		}
		return nil, fmt.Errorf("no routes found for node %s in model %s", nodeID, modelID)
	}
	
	return nil, fmt.Errorf("no routing table exists for model %s", modelID)
}

// GetRouteParameters returns the routing parameters between two nodes for a model
func (rt *RoutingTable) GetRouteParameters(modelID, sourceNodeID, destNodeID string) (RouteParameters, error) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	
	if modelParams, ok := rt.routeParams[modelID]; ok {
		if sourceParams, ok := modelParams[sourceNodeID]; ok {
			if params, ok := sourceParams[destNodeID]; ok {
				return params, nil
			}
		}
	}
	
	return RouteParameters{}, fmt.Errorf("no route parameters found for path %s→%s in model %s", 
		sourceNodeID, destNodeID, modelID)
}

// SetRouteParameters updates routing parameters for a specific route
func (rt *RoutingTable) SetRouteParameters(modelID, sourceNodeID, destNodeID string, params RouteParameters) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	
	// Verify that the route exists
	if modelRoutes, ok := rt.routes[modelID]; ok {
		if nodeRoutes, ok := modelRoutes[sourceNodeID]; ok {
			routeExists := false
			for _, route := range nodeRoutes {
				if route == destNodeID {
					routeExists = true
					break
				}
			}
			
			if !routeExists {
				return fmt.Errorf("no route exists from %s to %s for model %s", 
					sourceNodeID, destNodeID, modelID)
			}
			
			// Initialize the params map structure if it doesn't exist
			if _, ok := rt.routeParams[modelID]; !ok {
				rt.routeParams[modelID] = make(map[string]map[string]RouteParameters)
			}
			
			if _, ok := rt.routeParams[modelID][sourceNodeID]; !ok {
				rt.routeParams[modelID][sourceNodeID] = make(map[string]RouteParameters)
			}
			
			// Update the parameters
			rt.routeParams[modelID][sourceNodeID][destNodeID] = params
			return nil
		}
	}
	
	return fmt.Errorf("route not found for model %s, source %s", modelID, sourceNodeID)
}

// UpdateRoutesAfterNodeChange rebuilds routes after node topology changes
func (rt *RoutingTable) UpdateRoutesAfterNodeChange(modelID string) error {
	// Simply rebuild the entire routing table for the model
	return rt.BuildRoutes(modelID)
}

// OptimizeRoutes attempts to optimize the routing table for better performance
func (rt *RoutingTable) OptimizeRoutes(modelID string) error {
	// This is a placeholder for route optimization logic
	// In a real implementation, this would analyze network topology, bandwidth,
	// and compute resources to determine optimal routes
	
	fmt.Printf("Optimizing routes for model %s\n", modelID)
	
	rt.mu.Lock()
	defer rt.mu.Unlock()
	
	if _, exists := rt.routes[modelID]; !exists {
		return fmt.Errorf("no routing table exists for model %s", modelID)
	}
	
	// For now, we'll just adjust some parameters for existing routes
	for sourceID, destinations := range rt.routes[modelID] {
		for _, destID := range destinations {
			if params, ok := rt.routeParams[modelID][sourceID][destID]; ok {
				// Increase compression for routes with low bandwidth
				if params.Bandwidth < 500 {
					params.CompressionLevel = 9 // Higher compression
					rt.routeParams[modelID][sourceID][destID] = params
				}
				
				// Increase priority for routes with high latency
				if params.Latency > 10 {
					params.Priority = params.Priority + 2
					rt.routeParams[modelID][sourceID][destID] = params
				}
			}
		}
	}
	
	return nil
}

// DetectNetworkBottlenecks finds potential network bottlenecks in the routing
func (rt *RoutingTable) DetectNetworkBottlenecks(modelID string) ([]string, error) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	
	var bottlenecks []string
	
	// Get all nodes involved in this model
	nodesMap := make(map[string]bool)
	if modelRoutes, ok := rt.routes[modelID]; ok {
		for sourceID, destinations := range modelRoutes {
			nodesMap[sourceID] = true
			for _, destID := range destinations {
				nodesMap[destID] = true
			}
		}
	} else {
		return nil, fmt.Errorf("no routing table exists for model %s", modelID)
	}
	
	// For each node, count the number of connections
	for nodeID := range nodesMap {
		incomingCount := 0
		outgoingCount := 0
		
		// Count outgoing connections
		if outRoutes, ok := rt.routes[modelID][nodeID]; ok {
			outgoingCount = len(outRoutes)
		}
		
		// Count incoming connections
		for _, destinations := range rt.routes[modelID] {
			for _, destID := range destinations {
				if destID == nodeID {
					incomingCount++
					break
				}
			}
		}
		
		// Detect bottlenecks (nodes with many connections)
		totalConnections := incomingCount + outgoingCount
		if totalConnections > 5 { // Arbitrary threshold for illustration
			bottlenecks = append(bottlenecks, fmt.Sprintf(
				"Node %s has %d connections (%d in, %d out) and may be a network bottleneck",
				nodeID, totalConnections, incomingCount, outgoingCount))
		}
	}
	
	return bottlenecks, nil
}
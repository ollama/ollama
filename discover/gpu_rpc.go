package discover

import (
	"encoding/binary"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

// getRPCDeviceCount gets the number of devices available on an RPC server
func getRPCDeviceCount(endpoint string) uint32 {
	timeout := time.Duration(5 * 1000 * 1000 * 1000)

	client, err := net.DialTimeout("tcp", endpoint, timeout)
	if err != nil {
		return 0
	}
	defer client.Close()

	// First send RPC_CMD_HELLO (14)
	client.SetDeadline(time.Now().Add(timeout))
	if _, err := client.Write([]byte{14}); err != nil {
		return 0
	}

	// HELLO input size (8 bytes) - no input
	client.SetDeadline(time.Now().Add(timeout))
	helloSize := [8]byte{}
	if _, err := client.Write(helloSize[:]); err != nil {
		return 0
	}

	// Read HELLO reply size (8 bytes) - should be 3
	client.SetDeadline(time.Now().Add(timeout))
	helloReplySize := [8]byte{}
	if _, err := client.Read(helloReplySize[:]); err != nil {
		return 0
	}

	// Read HELLO reply (3 bytes - version)
	client.SetDeadline(time.Now().Add(timeout))
	serverVersion := [3]byte{}
	if _, err := client.Read(serverVersion[:]); err != nil {
		return 0
	}

	// Now send RPC_CMD_DEVICE_COUNT command (15)
	client.SetDeadline(time.Now().Add(timeout))
	if _, err := client.Write([]byte{15}); err != nil {
		return 0
	}

	// Input Size (8 bytes) - no input needed
	client.SetDeadline(time.Now().Add(timeout))
	inputSize := [8]byte{}
	if _, err := client.Write(inputSize[:]); err != nil {
		return 0
	}

	// Read reply size (8 bytes)
	client.SetDeadline(time.Now().Add(timeout))
	replySize := [8]byte{}
	if _, err := client.Read(replySize[:]); err != nil {
		return 0
	}

	// Read device count (4 bytes)
	client.SetDeadline(time.Now().Add(timeout))
	deviceCountBytes := [4]byte{}
	if _, err := client.Read(deviceCountBytes[:]); err != nil {
		return 0
	}

	return binary.LittleEndian.Uint32(deviceCountBytes[:])
}

// getRPCDeviceMemory checks total and free memory in bytes of a specific device
// on a given RPC endpoint.
//
// If the RPC endpoint given is unavailable (unable to connect), the total and
// free memory returned would be 0.
func getRPCDeviceMemory(endpoint string, deviceIndex uint32) RPCServerMemoryResult {
	// Setting timeout to 5 seconds
	var deadLine time.Time
	timeout := time.Duration(5 * 1000 * 1000 * 1000)

	slog.Debug("getting memory for RPC server", "endpoint", endpoint)
	// Creating RPC client
	client, err := net.DialTimeout("tcp", endpoint, timeout)
	if err != nil {
		return RPCServerMemoryResult{}
	}
	defer client.Close()
	slog.Debug("connection established with server", "endpoint", endpoint)

	// Sending RPC_CMD_HELLO command
	// RPC Command (1 byte)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	_, err = client.Write([]byte{14})
	if err != nil {
		slog.Error("failed to send RPC_CMD_HELLO command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent RPC_CMD_HELLO command")
	// Input Size (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	helloSize := [8]byte{}
	_, err = client.Write(helloSize[:])
	if err != nil {
		slog.Error("failed to send input size of RPC_CMD_HELLO command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent input size of RPC_CMD_HELLO command")

	// Retrieving results for RPC_CMD_HELLO command
	// Getting reply size (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	helloReply := [8]byte{}
	_, err = client.Read(helloReply[:])
	if err != nil {
		slog.Error("failed to fetch RPC server reply size of RPC_CMD_HELLO", "err", err)
		return RPCServerMemoryResult{}
	}
	helloReplySize := binary.LittleEndian.Uint64(helloReply[:])
	slog.Debug("RPC_CMD_HELLO reply size", "size", helloReplySize)
	// Reply size should be 3 according to spec
	if helloReplySize != 3 {
		slog.Error("invalid reply size for RPC_CMD_HELLO")
		return RPCServerMemoryResult{}
	}
	// Getting main reply
	// The version of the RPC server (3 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	serverVersion := [3]byte{}
	_, err = client.Read(serverVersion[:])
	if err != nil {
		slog.Error("failed to fetch RPC server version from RPC_CMD_HELLO", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("RPC_CMD_HELLO reply", "major", serverVersion[0], "minor", serverVersion[1], "patch", serverVersion[2])

	// Sending RPC_CMD_GET_DEVICE_MEMORY command
	// RPC Command (1 byte)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	_, err = client.Write([]byte{11})
	if err != nil {
		slog.Error("failed to send RPC_CMD_GET_DEVICE_MEMORY command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent RPC_CMD_GET_DEVICE_MEMORY command")
	// Input Size (8 bytes) - must be 4 bytes (sizeof(uint32_t) for device ID)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	size := [8]byte{}
	binary.LittleEndian.PutUint64(size[:], 4) // Input is 4 bytes (device ID)
	_, err = client.Write(size[:])
	if err != nil {
		slog.Error("failed to send input size of RPC_CMD_GET_DEVICE_MEMORY command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent input size RPC_CMD_GET_DEVICE_MEMORY command")
	// Device ID (4 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	deviceID := [4]byte{}
	binary.LittleEndian.PutUint32(deviceID[:], deviceIndex)
	_, err = client.Write(deviceID[:])
	if err != nil {
		slog.Error("failed to send device ID for RPC_CMD_GET_DEVICE_MEMORY command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent device ID for RPC_CMD_GET_DEVICE_MEMORY command")

	// Retrieving results for RPC_CMD_GET_DEVICE_MEMORY command
	// Getting reply size (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	reply := [8]byte{}
	_, err = client.Read(reply[:])
	if err != nil {
		slog.Error("failed to fetch RPC server reply size of RPC_CMD_GET_DEVICE_MEMORY", "err", err)
		return RPCServerMemoryResult{}
	}
	reply_size := binary.LittleEndian.Uint64(reply[:])
	// Reply size should be 16 according to spec
	if reply_size != 16 {
		slog.Error("invalid reply size received from RPC server")
		return RPCServerMemoryResult{}
	}
	// Getting main reply
	// The free memory of the RPC server (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	freeMem := [8]byte{}
	_, err = client.Read(freeMem[:])
	if err != nil {
		return RPCServerMemoryResult{}
	}
	// The total memory of the RPC server (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	totalMem := [8]byte{}
	_, err = client.Read(totalMem[:])
	if err != nil {
		return RPCServerMemoryResult{}
	}

	return RPCServerMemoryResult{
		FreeMem:  binary.LittleEndian.Uint64(freeMem[:]),
		TotalMem: binary.LittleEndian.Uint64(totalMem[:]),
	}
}

// Find valid RPC servers from a comma seperated list of endpoints.
func GetRPCServers(endpoints string) []ml.DeviceInfo {
	slog.Debug("finding valid rpc servers", "endpoints", endpoints)
	rpcServersList := strings.Split(endpoints, ",")
	var validServers []ml.DeviceInfo
	for _, server := range rpcServersList {
		// No servers given
		if server == "" {
			continue
		}

		// Trim whitespace
		server = strings.TrimSpace(server)

		// Validate server address
		serverAddress := strings.Split(server, ":")
		if len(serverAddress) != 2 {
			slog.Warn("invalid RPC endpoint server address", "endpoint", server)
			continue
		}
		// Invalid port number
		_, err := strconv.ParseUint(serverAddress[len(serverAddress)-1], 10, 16)
		if err != nil {
			slog.Warn("invalid RPC endpoint port number", "endpoint", server)
			continue
		}

		// Get device count
		deviceCount := getRPCDeviceCount(server)
		if deviceCount == 0 {
			slog.Warn("unable to connect to endpoint or no devices found", "endpoint", server)
			continue
		}

		slog.Info("found RPC server", "endpoint", server, "device_count", deviceCount)

		// Enumerate all devices on this server
		for deviceIdx := uint32(0); deviceIdx < deviceCount; deviceIdx++ {
			info := getRPCDeviceMemory(server, deviceIdx)

			// Device ID format is endpoint:device_index (e.g., "127.0.0.1:50053:0")
			deviceID := server + ":" + strconv.FormatUint(uint64(deviceIdx), 10)

			serverInfo := ml.DeviceInfo{
				DeviceID: ml.DeviceID{
					ID:      deviceID,
					Library: "rpc",
				},
				TotalMemory: info.TotalMem,
				FreeMemory:  info.FreeMem,
			}

			if serverInfo.TotalMemory == 0 && serverInfo.FreeMemory == 0 {
				slog.Warn("unable to get memory for device", "endpoint", server, "device", deviceIdx)
			} else {
				slog.Debug("found RPC device", "id", deviceID, "total", serverInfo.TotalMemory, "free", serverInfo.FreeMemory)
				validServers = append(validServers, serverInfo)
			}
		}
	}

	return validServers
}

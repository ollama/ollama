package discover

import (
	"encoding/binary"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"time"
)

// RPCServerMemory checks and total and free memory in bytes of a given RPC
// endpoint.
//
// If the RPC endpoint given is unavailable (unable to connect), the total and
// free memory returned would be 0.
func getRPCServerMemory(endpoint string) RPCServerMemoryResult {
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
	// Input Size (8 bytes)
	deadLine = time.Now().Add(timeout)
	client.SetDeadline(deadLine)
	size := [8]byte{}
	_, err = client.Write(size[:])
	if err != nil {
		slog.Error("failed to send input size of RPC_CMD_GET_DEVICE_MEMORY command to RPC server", "err", err)
		return RPCServerMemoryResult{}
	}
	slog.Debug("successfully sent input size RPC_CMD_GET_DEVICE_MEMORY command")

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
func GetRPCServers(endpoints string) GpuInfoList {
	slog.Debug("finding valid rpc servers", "endpoints", endpoints)
	rpcServersList := strings.Split(endpoints, ",")
	var validServers GpuInfoList
	for _, server := range rpcServersList {
		// No servers given
		if server == "" {
			break
		}

		// Getting information
		info := getRPCServerMemory(server)
		serverAddress := strings.Split(server, ":")
		// We got an invalid server address
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

		serverInfo := GpuInfo{
			ID:      server,
			Library: "rpc",
		}

		serverInfo.TotalMemory = info.TotalMem
		serverInfo.FreeMemory = info.FreeMem

		if serverInfo.TotalMemory == 0 && serverInfo.FreeMemory == 0 {
			slog.Warn("unable to connect to endpoint", "endpoint", server)
		} else {
			slog.Debug("found RPC server", "info", serverInfo)
			validServers = append(validServers, serverInfo)
		}
	}

	return validServers
}

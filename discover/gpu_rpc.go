package discover

// RPC device discovery — adapted from PR ollama/ollama#10844 by @gkpln3.
//
// The main Ollama scheduler enumerates GPUs via subprocess bootstrap, which
// never sees devices registered by ggml_backend_rpc_add_server() in the runner.
// Instead of trying to plumb CGo into discover, we speak the ggml-rpc wire
// protocol directly over TCP and synthesize ml.DeviceInfo entries that look
// just like local GPUs to the scheduler.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

const (
	rpcCmdGetDeviceMemory byte = 11
	rpcCmdHello           byte = 14
	rpcDialTimeout             = 5 * time.Second
)

// rpcServerMemory queries free and total memory of a remote ggml-rpc server.
// Returns (0, 0) if the server is unreachable or speaks a different protocol.
func rpcServerMemory(endpoint string) (free, total uint64, err error) {
	conn, err := net.DialTimeout("tcp", endpoint, rpcDialTimeout)
	if err != nil {
		return 0, 0, fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	// Handshake: RPC_CMD_HELLO. Response is 3 bytes (major, minor, patch).
	if err := rpcCall(conn, rpcCmdHello, nil, 3); err != nil {
		return 0, 0, fmt.Errorf("hello: %w", err)
	}
	version := make([]byte, 3)
	if _, err := conn.Read(version); err != nil {
		return 0, 0, fmt.Errorf("read version: %w", err)
	}
	slog.Debug("rpc server version", "endpoint", endpoint, "version", fmt.Sprintf("%d.%d.%d", version[0], version[1], version[2]))

	// Query device 0 memory. Response is 16 bytes: free (u64) + total (u64).
	if err := rpcCall(conn, rpcCmdGetDeviceMemory, nil, 16); err != nil {
		return 0, 0, fmt.Errorf("get_device_memory: %w", err)
	}
	buf := make([]byte, 16)
	if _, err := conn.Read(buf); err != nil {
		return 0, 0, fmt.Errorf("read memory: %w", err)
	}
	return binary.LittleEndian.Uint64(buf[:8]),
		binary.LittleEndian.Uint64(buf[8:]),
		nil
}

// rpcCall sends a command byte + payload, then reads & validates the response
// size header (8 bytes LE), expecting expectedReplySize bytes.
func rpcCall(conn net.Conn, cmd byte, payload []byte, expectedReplySize uint64) error {
	conn.SetDeadline(time.Now().Add(rpcDialTimeout))
	if _, err := conn.Write([]byte{cmd}); err != nil {
		return err
	}
	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(payload)))
	if _, err := conn.Write(sizeBuf[:]); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := conn.Write(payload); err != nil {
			return err
		}
	}
	conn.SetDeadline(time.Now().Add(rpcDialTimeout))
	var replyBuf [8]byte
	if _, err := conn.Read(replyBuf[:]); err != nil {
		return err
	}
	if got := binary.LittleEndian.Uint64(replyBuf[:]); got != expectedReplySize {
		return fmt.Errorf("unexpected reply size %d (want %d)", got, expectedReplySize)
	}
	return nil
}

// discoverRPCDevices probes each endpoint in OLLAMA_RPC_SERVERS and returns
// synthetic DeviceInfo entries the scheduler can route layers to. The IDs are
// constructed to match what ggml-rpc names its devices on the runner side
// (Library="RPC", ID="RPC0[host:port]") so DeviceID lookups in the load
// handler resolve correctly.
func discoverRPCDevices(endpoints []string) []ml.DeviceInfo {
	var devices []ml.DeviceInfo
	for _, ep := range endpoints {
		ep = strings.TrimSpace(ep)
		if ep == "" {
			continue
		}
		host, port, err := net.SplitHostPort(ep)
		if err != nil {
			slog.Warn("invalid RPC endpoint, expected host:port", "endpoint", ep, "err", err)
			continue
		}
		if _, err := strconv.ParseUint(port, 10, 16); err != nil {
			slog.Warn("invalid RPC endpoint port", "endpoint", ep)
			continue
		}

		free, total, err := rpcServerMemory(ep)
		if err != nil || total == 0 {
			slog.Warn("RPC server unreachable", "endpoint", ep, "err", err)
			continue
		}

		devices = append(devices, ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				ID:      fmt.Sprintf("RPC0[%s]", ep),
				Library: "RPC",
			},
			Name:        "RPC",
			Description: fmt.Sprintf("remote ggml-rpc server at %s", ep),
			TotalMemory: total,
			FreeMemory:  free,
		})
		slog.Info("discovered RPC device", "endpoint", ep, "host", host, "free", free, "total", total)
	}
	return devices
}

// errNoRPCServers is returned from RPCDevices when OLLAMA_RPC_SERVERS is unset
// or contains only empty entries.
var errNoRPCServers = errors.New("no RPC servers configured")

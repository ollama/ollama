package tools

import (
	"fmt"
	"net"
	"net/netip"
	"strings"
	"sync"
)

type IPAllowList struct {
	allowList string
	cidrs     []*net.IPNet
	ips       []net.IP
	mu        sync.RWMutex
	enabled   bool
}

func NewIPAllowList(allowList string) (*IPAllowList, error) {

	w := &IPAllowList{}
	err := w.Update(allowList)
	return w, err
}

func (w *IPAllowList) GetAllowList() string {
	return w.allowList
}

func (w *IPAllowList) Update(allowListStr string) error {
	var cidrs []*net.IPNet
	var ips []net.IP

	allowList := make([]string, 0)
	if allowListStr != "" {
		allowList = strings.Split(allowListStr, ",")
	}

	for _, item := range allowList {
		_, cidrNet, err := net.ParseCIDR(item)
		if err == nil {
			cidrs = append(cidrs, cidrNet)
		} else {
			ip := net.ParseIP(item)
			if ip != nil {
				ips = append(ips, ip)
			} else {
				return fmt.Errorf("invalid allowList item: %s", item)
			}
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	w.allowList = allowListStr
	w.cidrs = cidrs
	w.ips = ips
	w.enabled = len(cidrs) > 0 || len(ips) > 0
	return nil
}

func (w *IPAllowList) IsAllowed(ip interface{}) bool {
	if !w.enabled {
		return true
	}

	var parsedIP net.IP
	switch v := ip.(type) {
	case string:
		parsedIP = net.ParseIP(v)
	case net.IP:
		parsedIP = v
	case netip.Addr:
		parsedIP = net.IP(v.AsSlice())
	default:
		if str, ok := v.(string); ok {
			parsedIP = net.ParseIP(str)
		}
	}

	if parsedIP == nil {
		return false
	}

	w.mu.RLock()
	defer w.mu.RUnlock()

	for _, cidr := range w.cidrs {
		if cidr.Contains(parsedIP) {
			return true
		}
	}

	for _, allowedIP := range w.ips {
		if parsedIP.Equal(allowedIP) {
			return true
		}
	}
	return false
}

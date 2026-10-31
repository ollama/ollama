package transfer

import (
	"context"
	"errors"
	"net"
	"slices"
	"syscall"
	"testing"
)

func TestIsPublicIPNAT64(t *testing.T) {
	for _, tc := range []struct {
		ipv4   string
		public bool
	}{
		{"13.39.241.58", true},
		{"8.8.8.8", true},
		{"0.0.0.0", false},
		{"0.0.0.1", false},
		{"10.0.0.1", false},
		{"100.100.2.148", false},
		{"127.0.0.1", false},
		{"169.254.169.254", false},
		{"172.16.0.1", false},
		{"192.0.0.8", false},
		{"192.0.2.1", false},
		{"192.88.99.1", false},
		{"192.168.0.1", false},
		{"198.18.0.1", false},
		{"198.51.100.1", false},
		{"203.0.113.1", false},
		{"224.0.0.1", false},
		{"240.0.0.1", false},
		{"255.255.255.255", false},
	} {
		t.Run(tc.ipv4, func(t *testing.T) {
			ip := net.ParseIP("64:ff9b::")
			copy(ip[12:], net.ParseIP(tc.ipv4).To4())
			if got := isPublicIP(ip); got != tc.public {
				t.Errorf("isPublicIP(%s) = %t, want %t", ip, got, tc.public)
			}
		})
	}
	for _, address := range []string{"64:ff9b:1::808:808", "64:ff9b::1:808:808"} {
		if isPublicIP(net.ParseIP(address)) {
			t.Errorf("non-well-known NAT64 prefix %s allowed", address)
		}
	}
}

func TestPublicRedirectIPsDNS64(t *testing.T) {
	original := lookupRedirectIPs
	t.Cleanup(func() { lookupRedirectIPs = original })
	for _, tc := range []struct {
		name    string
		answers []net.IP
		wantErr bool
	}{
		{"IPv6 only", []net.IP{net.ParseIP("64:ff9b::d27:f13a")}, false},
		{"dual stack", []net.IP{net.ParseIP("13.39.241.58"), net.ParseIP("64:ff9b::d27:f13a")}, false},
		{"mixed public and metadata", []net.IP{net.ParseIP("13.39.241.58"), net.ParseIP("64:ff9b::a9fe:a9fe")}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lookupRedirectIPs = func(context.Context, string, string) ([]net.IP, error) {
				return tc.answers, nil
			}
			ips, err := publicRedirectIPs(t.Context(), "cdn.example")
			if tc.wantErr {
				if !errors.Is(err, errRedirectNotAllowed) {
					t.Fatalf("expected redirect rejection, got %v", err)
				}
			} else if err != nil || !slices.EqualFunc(ips, tc.answers, net.IP.Equal) {
				t.Fatalf("got %v, %v; want unchanged DNS answers %v", ips, err, tc.answers)
			}
		})
	}
}

func TestCheckedDialerNAT64(t *testing.T) {
	stopped := errors.New("stop before connecting")
	wantAddr := "[64:ff9b::d27:f13a]:443"
	dial := checkedDialer(&net.Dialer{
		Control: func(network, address string, _ syscall.RawConn) error {
			if network != "tcp6" || address != wantAddr {
				t.Errorf("dialing %s %s, want tcp6 %s", network, address, wantAddr)
			}
			return stopped
		},
	}, false)
	if _, err := dial(t.Context(), "tcp", wantAddr); !errors.Is(err, stopped) {
		t.Fatalf("public NAT64 target did not reach dial: %v", err)
	}
	if _, err := dial(t.Context(), "tcp", "[64:ff9b::a9fe:a9fe]:443"); !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("metadata NAT64 target was not rejected before dial: %v", err)
	}
}

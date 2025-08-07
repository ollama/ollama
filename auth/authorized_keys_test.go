package auth

import (
	"bytes"
	"reflect"
	"testing"
)

const validB64 = "AAAAC3NzaC1lZDI1NTE5AAAAICy1v/Sn0kGhu1LXzCsnx3wlk5ESdncS66JWo13yeJod"

func TestParse(t *testing.T) {
	tests := []struct {
		name string
		file string
		want map[string]*KeyPermission
	}{
		{
			name: "two fields only defaults",
			file: "ssh-ed25519 " + validB64 + "\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "default",
					Endpoints: []string{"*"},
				},
			},
		},
		{
			name: "extra whitespace collapsed and default endpoints",
			file: "ssh-ed25519  " + validB64 + "   alice\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "alice",
					Endpoints: []string{"*"},
				},
			},
		},
		{
			name: "four fields full",
			file: "ssh-ed25519 " + validB64 + " bob /api/foo,/api/bar\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "bob",
					Endpoints: []string{"/api/foo", "/api/bar"},
				},
			},
		},
		{
			name: "comment lines ignored and multiple entries",
			file: "# header\n\nssh-ed25519 " + validB64 + " user1\nssh-ed25519 " + validB64 + "  user2  /api/x\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "user1",
					Endpoints: []string{"*"},
				},
			},
		},
		{
			name: "three entries variety",
			file: "ssh-ed25519 " + validB64 + "\nssh-ed25519 " + validB64 + " alice /api/a,/api/b\nssh-ed25519 " + validB64 + " bob /api/c\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "alice",
					Endpoints: []string{"*"},
				},
			},
		},
		{
			name: "two entries w/ wildcard",
			file: "ssh-ed25519 " + validB64 + " alice /api/a\n* * * /api/b\n",
			want: map[string]*KeyPermission{
				validB64: {
					Name:      "alice",
					Endpoints: []string{"/api/a"},
				},
				"*": {
					Name:      "default",
					Endpoints: []string{"/api/b"},
				},
			},
		},
		{
			name: "tags for everyone",
			file: "* * * /api/tags",
			want: map[string]*KeyPermission{
				"*": {
					Name:      "default",
					Endpoints: []string{"/api/tags"},
				},
			},
		},
		{
			name: "default name",
			file: "* * somename",
			want: map[string]*KeyPermission{
				"*": {
					Name:      "somename",
					Endpoints: []string{"*"},
				},
			},
		},
		{
			name: "unsupported key type",
			file: "ssh-rsa AAAAB3Nza...\n",
			want: map[string]*KeyPermission{},
		},
		{
			name: "bad base64",
			file: "ssh-ed25519 invalid@@@\n",
			want: map[string]*KeyPermission{},
		},
		{
			name: "just an asterix",
			file: "*\n",
			want: map[string]*KeyPermission{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			perms := NewAPIPermissions()
			err := perms.parse(bytes.NewBufferString(tc.file))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(perms.permissions) != len(tc.want) {
				t.Fatalf("got %d entries, want %d", len(perms.permissions), len(tc.want))
			}
			if !reflect.DeepEqual(perms.permissions, tc.want) {
				t.Errorf("got %+v, want %+v", perms.permissions, tc.want)
			}
		})
	}
}

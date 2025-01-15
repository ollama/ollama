package parser

import (
	"os"
	"os/user"
	"path/filepath"
	"testing"
)

func TestExpandPath(t *testing.T) {
	mockCurrentUser := func() (*user.User, error) {
		return &user.User{
			Username: "testuser",
			HomeDir: func() string {
				if os.PathSeparator == '\\' {
					return filepath.FromSlash("D:/home/testuser")
				}
				return "/home/testuser"
			}(),
		}, nil
	}

	mockLookupUser := func(username string) (*user.User, error) {
		fakeUsers := map[string]string{
			"testuser": func() string {
				if os.PathSeparator == '\\' {
					return filepath.FromSlash("D:/home/testuser")
				}
				return "/home/testuser"
			}(),
			"anotheruser": func() string {
				if os.PathSeparator == '\\' {
					return filepath.FromSlash("D:/home/anotheruser")
				}
				return "/home/anotheruser"
			}(),
		}

		if homeDir, ok := fakeUsers[username]; ok {
			return &user.User{
				Username: username,
				HomeDir:  homeDir,
			}, nil
		}
		return nil, os.ErrNotExist
	}

	tests := []struct {
		path            string
		relativeDir     string
		expected        string
		windowsExpected string
		shouldErr       bool
	}{
		{"~", "", "/home/testuser", "D:\\home\\testuser", false},
		{"~/myfolder/myfile.txt", "", "/home/testuser/myfolder/myfile.txt", "D:\\home\\testuser\\myfolder\\myfile.txt", false},
		{"~anotheruser/docs/file.txt", "", "/home/anotheruser/docs/file.txt", "D:\\home\\anotheruser\\docs\\file.txt", false},
		{"~nonexistentuser/file.txt", "", "", "", true},
		{"relative/path/to/file", "", filepath.Join(os.Getenv("PWD"), "relative/path/to/file"), "relative\\path\\to\\file", false},
		{"/absolute/path/to/file", "", "/absolute/path/to/file", "D:\\absolute\\path\\to\\file", false},
		{"/absolute/path/to/file", "someotherdir/", "/absolute/path/to/file", "D:\\absolute\\path\\to\\file", false},
		{".", os.Getenv("PWD"), os.Getenv("PWD"), os.Getenv("PWD"), false},
		{".", "", os.Getenv("PWD"), os.Getenv("PWD"), false},
		{"somefile", "somedir", filepath.Join(os.Getenv("PWD"), "somedir", "somefile"), "somedir\\somefile", false},
	}

	for _, test := range tests {
		result, err := expandPathImpl(test.path, test.relativeDir, mockCurrentUser, mockLookupUser)
		if (err != nil) != test.shouldErr {
			t.Errorf("expandPathImpl(%q) returned error: %v, expected error: %v", test.path, err != nil, test.shouldErr)
		}

		if os.PathSeparator == '\\' {
			if result != test.windowsExpected && !test.shouldErr {
				t.Errorf("(win) expandPathImpl(%q) = %q, want %q", test.path, result, test.windowsExpected)
			}
		} else {
			if result != test.expected && !test.shouldErr {
				t.Errorf("expandPathImpl(%q) = %q, want %q", test.path, result, test.expected)
			}
		}
	}
}

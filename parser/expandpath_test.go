package parser

import (
	"os"
	"os/user"
	"path/filepath"
	"runtime"
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

	pwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	t.Run("unix tests", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			return
		}

		tests := []struct {
			path        string
			relativeDir string
			expected    string
			shouldErr   bool
		}{
			{"~", "", "/home/testuser", false},
			{"~/myfolder/myfile.txt", "", "/home/testuser/myfolder/myfile.txt", false},
			{"~anotheruser/docs/file.txt", "", "/home/anotheruser/docs/file.txt", false},
			{"~nonexistentuser/file.txt", "", "", true},
			{"relative/path/to/file", "", filepath.Join(pwd, "relative/path/to/file"), false},
			{"/absolute/path/to/file", "", "/absolute/path/to/file", false},
			{"/absolute/path/to/file", "someotherdir/", "/absolute/path/to/file", false},
			{".", pwd, pwd, false},
			{".", "", pwd, false},
			{"somefile", "somedir", filepath.Join(pwd, "somedir", "somefile"), false},
		}

		for _, test := range tests {
			result, err := expandPathImpl(test.path, test.relativeDir, mockCurrentUser, mockLookupUser)
			if (err != nil) != test.shouldErr {
				t.Errorf("expandPathImpl(%q) returned error: %v, expected error: %v", test.path, err != nil, test.shouldErr)
			}

			if result != test.expected && !test.shouldErr {
				t.Errorf("expandPathImpl(%q) = %q, want %q", test.path, result, test.expected)
			}
		}
	})

	t.Run("windows tests", func(t *testing.T) {
		if runtime.GOOS != "windows" {
			return
		}

		tests := []struct {
			path        string
			relativeDir string
			expected    string
			shouldErr   bool
		}{
			{"~", "", "D:\\home\\testuser", false},
			{"~/myfolder/myfile.txt", "", "D:\\home\\testuser\\myfolder\\myfile.txt", false},
			{"~anotheruser/docs/file.txt", "", "D:\\home\\anotheruser\\docs\\file.txt", false},
			{"~nonexistentuser/file.txt", "", "", true},
			{"relative\\path\\to\\file", "", filepath.Join(pwd, "relative\\path\\to\\file"), false},
			{"D:\\absolute\\path\\to\\file", "", "D:\\absolute\\path\\to\\file", false},
			{"D:\\absolute\\path\\to\\file", "someotherdir/", "D:\\absolute\\path\\to\\file", false},
			{".", pwd, pwd, false},
			{".", "", pwd, false},
			{"somefile", "somedir", filepath.Join(pwd, "somedir", "somefile"), false},
		}

		for _, test := range tests {
			result, err := expandPathImpl(test.path, test.relativeDir, mockCurrentUser, mockLookupUser)
			if (err != nil) != test.shouldErr {
				t.Errorf("expandPathImpl(%q) returned error: %v, expected error: %v", test.path, err != nil, test.shouldErr)
			}

			if result != test.expected && !test.shouldErr {
				t.Errorf("expandPathImpl(%q) = %q, want %q", test.path, result, test.expected)
			}
		}
	})
}

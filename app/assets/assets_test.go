package assets

import (
    "testing"
    "testing/fstest"
    "bytes"
)


// Test generated using Keploy
func TestGetIcon_ReturnsIconData(t *testing.T) {
    // Setup: use an in-memory FS with some files
    Icons = fstest.MapFS{
        "icon1.ico": &fstest.MapFile{Data: []byte{0x00, 0x01, 0x02}},
    }

    // Call the function
    data, err := GetIcon("icon1.ico")
    if err != nil {
        t.Fatalf("Expected no error, got %v", err)
    }

    expectedData := []byte{0x00, 0x01, 0x02}
    if !bytes.Equal(data, expectedData) {
        t.Errorf("Expected %v, got %v", expectedData, data)
    }
}

// Test generated using Keploy
func TestListIcons_MultipleIcons(t *testing.T) {
    // Setup: use an in-memory FS with multiple .ico files
    Icons = fstest.MapFS{
        "icon1.ico": &fstest.MapFile{Data: []byte("icon1 data")},
        "icon2.ico": &fstest.MapFile{Data: []byte("icon2 data")},
        "icon3.ico": &fstest.MapFile{Data: []byte("icon3 data")},
    }

    // Call the function
    iconsList, err := ListIcons()
    if err != nil {
        t.Fatalf("Expected no error, got %v", err)
    }

    // Expected icons
    expected := []string{"icon1.ico", "icon2.ico", "icon3.ico"}

    // Since fs.Glob does not guarantee order, we can use a map to compare
    expectedMap := map[string]bool{}
    for _, e := range expected {
        expectedMap[e] = true
    }

    for _, name := range iconsList {
        if !expectedMap[name] {
            t.Errorf("Unexpected icon name: %s", name)
        }
        delete(expectedMap, name)
    }

    if len(expectedMap) != 0 {
        t.Errorf("Missing icons: %v", expectedMap)
    }
}


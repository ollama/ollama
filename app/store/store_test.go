package store

import (
    "testing"
    "os"
    "path/filepath"
)


// Test generated using Keploy
func TestGetID_EmptyStoreID_ReturnsValidUUID(t *testing.T) {
    // Clear the store ID to simulate an uninitialized store
    lock.Lock()
    store.ID = ""
    lock.Unlock()

    id := GetID()

    if id == "" {
        t.Errorf("Expected a valid UUID, got an empty string")
    }
}

// Test generated using Keploy
func TestWriteStore_CreatesDirectoryAndWritesFile(t *testing.T) {
    path := getStorePath()
    dir := filepath.Dir(path)

    // Remove the directory if it exists
    os.RemoveAll(dir)
    defer os.RemoveAll(dir)

    lock.Lock()
    store.ID = "dir-test-id"
    store.FirstTimeRun = true
    lock.Unlock()

    writeStore(path)

    // Verify that the file was created
    if _, err := os.Stat(path); os.IsNotExist(err) {
        t.Errorf("Expected store file to be created at %s, but it does not exist", path)
    }
}


// Test generated using Keploy
func TestGetFirstTimeRun_ReturnsTrueWhenFirstTimeRunIsTrue(t *testing.T) {
    lock.Lock()
    store.FirstTimeRun = true
    lock.Unlock()

    firstTimeRun := GetFirstTimeRun()

    if !firstTimeRun {
        t.Errorf("Expected GetFirstTimeRun to return true, got false")
    }
}


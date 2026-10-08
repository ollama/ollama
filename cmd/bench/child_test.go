package main

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

// TestChildHelperProcess is not a test: it is the process the child tests start.
func TestChildHelperProcess(t *testing.T) {
	switch os.Getenv("BENCH_CHILD_HELPER") {
	case "exit":
		os.Exit(3)
	case "block":
		time.Sleep(time.Minute)
	}
}

func helperCommand(mode string) *exec.Cmd {
	cmd := exec.Command(os.Args[0], "-test.run=^TestChildHelperProcess$")
	cmd.Env = append(os.Environ(), "BENCH_CHILD_HELPER="+mode)
	return cmd
}

func TestChildDetectsEarlyExit(t *testing.T) {
	c, err := startChild(helperCommand("exit"))
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-c.exited:
	case <-time.After(30 * time.Second):
		t.Fatal("exit was not observed")
	}
	if c.alive() {
		t.Error("alive after exit")
	}
	c.stop(time.Second) // an exited child stops without waiting
}

func TestChildStopEndsRunningProcess(t *testing.T) {
	c, err := startChild(helperCommand("block"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.stop(time.Second) })
	if !c.alive() {
		t.Fatal("not alive after start")
	}
	c.stop(5 * time.Second)
	if c.alive() {
		t.Error("alive after stop")
	}
	c.stop(5 * time.Second) // second stop is a no-op
}

package main

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func scenarioTestOptions(t *testing.T, m *mockRunner, spec string) flagOptions {
	fOpt := directTestOptions(t, modeBoth, m)
	fOpt.scenario = &spec
	promptTokens := 3000
	fOpt.promptTokens = &promptTokens
	epochs := 2
	fOpt.epochs = &epochs
	return fOpt
}

func TestBenchmarkScenarios_WorkingCache(t *testing.T) {
	m := &mockRunner{prefix: true, coldLimit: 150_000}
	fOpt := scenarioTestOptions(t, m, "all")

	var out strings.Builder
	var err error
	captureOutput(func() { err = benchmarkScenarios(fOpt, &out) })
	if err != nil {
		t.Fatalf("scenarios failed: %v\n%s", err, out.String())
	}

	var wants []string
	for _, name := range scenarioNames() {
		wants = append(wants, "scenario="+name+"/step=")
	}
	for _, want := range wants {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "VOID") {
		t.Errorf("unexpected void epoch:\n%s", out.String())
	}
	for i, req := range m.requests {
		if !req.Stats || (!req.IgnoreEOS && req.Format == nil) {
			t.Errorf("request %d: Stats=%v IgnoreEOS=%v, want both set", i, req.Stats, req.IgnoreEOS)
		}
	}
}

func TestBenchmarkScenarios_BrokenCacheIsVoid(t *testing.T) {
	m := &mockRunner{} // reports zero cached tokens for every request
	fOpt := scenarioTestOptions(t, m, "repeat")

	var out strings.Builder
	var err error
	captureOutput(func() { err = benchmarkScenarios(fOpt, &out) })
	if err == nil {
		t.Fatal("expected an error when the cache never hits")
	}
	if !strings.Contains(out.String(), "# VOID scenario=repeat") {
		t.Errorf("expected VOID lines:\n%s", out.String())
	}
	if strings.Contains(out.String(), "BenchmarkCache/") {
		t.Errorf("void epochs must emit no metrics:\n%s", out.String())
	}
}

func TestSelectScenarios(t *testing.T) {
	if _, err := selectScenarios("repeat,nope"); err == nil {
		t.Error("expected an error for an unknown scenario")
	}
	got, err := selectScenarios("branch,repeat")
	if err != nil || len(got) != 2 || got[0].name != "branch" {
		t.Errorf("got %v, %v", got, err)
	}
}

func TestScenarioCancelStepRejectsFastFailure(t *testing.T) {
	m := &mockRunner{failWith: "boom"}
	fOpt := directTestOptions(t, modeBoth, m)
	b, err := newRunnerBackend(fOpt)
	if err != nil {
		t.Fatal(err)
	}
	s := &scenarioRun{backend: b, fOpt: fOpt, timeout: 10 * time.Second}
	_, err = s.step(stepSpec{name: "cancelled", prompt: "x", cancel: 10 * time.Second})
	if err == nil || errors.Is(err, errVoid) || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err = %v, want the runner's failure as a hard error", err)
	}
}

func TestScenarioMediaSkipsTextOnlyModels(t *testing.T) {
	m := &mockRunner{prefix: true, noMedia: true}
	fOpt := scenarioTestOptions(t, m, "media")
	var out strings.Builder
	var err error
	captureOutput(func() { err = benchmarkScenarios(fOpt, &out) })
	if err != nil {
		t.Fatalf("media on a text-only model failed: %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "# SKIP scenario=media") {
		t.Errorf("expected a SKIP line:\n%s", out.String())
	}
}

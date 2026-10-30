package launch

import "strings"

// OverrideIntegration replaces one registry entry's runner for tests and returns a restore function.
func OverrideIntegration(name string, runner Runner) func() {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		key := strings.ToLower(name)
		integrationSpecsByName[key] = &IntegrationSpec{Name: key, Runner: runner}
		return func() {
			delete(integrationSpecsByName, key)
		}
	}

	original := spec.Runner
	spec.Runner = runner
	return func() {
		spec.Runner = original
	}
}

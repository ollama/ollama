package convert

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"strings"

	"github.com/goccy/go-yaml"
	"github.com/ollama/ollama/fs/ggml"
)

// Custom type for YAML unmarshaling of model card metadata (as per HF YAML schema)
type StringOrSlice []string

// Actionable subset of the general HF model metadata (as per HF YAML schema) for operations
// See: https://huggingface.co/docs/hub/en/model-cards#editing-the-yaml-section-of-the-readmemd-file
type ModelCardMetadata struct {
	BaseModels  StringOrSlice `yaml:"base_model"`
	Datasets    StringOrSlice `yaml:"datasets"`
	License     string        `yaml:"license"`
	LicenseLink string        `yaml:"license_link"`
	LicenseName string        `yaml:"license_name"`
	Languages   []string      `yaml:"language"`
	Tags        []string      `yaml:"tags"`
}

// See: https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py
func (metadata *ModelCardMetadata) KV() (kv ggml.KV) {
	kv = make(ggml.KV)
	// Licensing data
	if metadata.License != "" {
		kv["general.license"] = metadata.License
	}
	if metadata.LicenseName != "" {
		kv["general.license.name"] = metadata.LicenseName
	}
	if metadata.LicenseLink != "" {
		kv["general.license.link"] = metadata.LicenseLink
	}
	// Model Provenance
	kv["general.base_model.count"] = uint32(len(metadata.BaseModels))
	for i, repo_id := range metadata.BaseModels {
		kv[fmt.Sprintf("general.base_model.%v.repo_url", i)] = "https://huggingface.co/" + repo_id
	}
	kv["general.dataset.count"] = uint32(len(metadata.BaseModels))
	for i, repo_id := range metadata.Datasets {
		kv[fmt.Sprintf("general.dataset.%v.repo_url", i)] = "https://huggingface.co/" + repo_id
	}
	// Languages
	if metadata.Languages != nil {
		languages, err := json.Marshal(metadata.Languages)
		if err == nil {
			if strLanguages := string(languages); strLanguages != "" {
				kv["general.languages"] = strLanguages
			}
		}
	}
	// Tags
	if metadata.Tags != nil {
		tags, err := json.Marshal(metadata.Tags)
		if err == nil {
			if strTags := string(tags); strTags != "" {
				kv["general.tags"] = strTags
			}
		}
	}
	return
}

func parseModelCardYAML(fsys fs.FS, modelCardFilename string) (data []byte, err error) {
	file, err := fsys.Open(modelCardFilename)
	if err != nil {
		slog.Warn(fmt.Sprintf("Unable to open file: %s", modelCardFilename))
		return
	}
	defer file.Close()

	yamlDelimiter := "---"
	scanner := bufio.NewScanner(file)

	// Affirm YAML metadata is at the top of the file by presence of HF delim.
	if !scanner.Scan() {
		slog.Warn(fmt.Sprintf("Unable to scan content of file: %s", modelCardFilename))
		err = scanner.Err()
		return
	}

	line := scanner.Text()
	if !strings.HasPrefix(line, yamlDelimiter) {
		slog.Warn(fmt.Sprintf("YAML metadata not found at top of file: %s", modelCardFilename))
		return
	}

	// Read raw YAML metadata until HF delim. encountered
	for scanner.Scan() {
		line = scanner.Text() + "\n"
		// stop scanning when yaml delim. found
		if strings.HasPrefix(line, yamlDelimiter) {
			break
		}
		data = append(data, []byte(line)...)
	}

	// Account for errors reading the file during for loop
	if err = scanner.Err(); err != nil {
		return
	}
	return
}

// Attempt to process model metadata if it exists
func unmarshalModelCardMetadata(yamlData []byte, metadata *ModelCardMetadata) (err error) {
	// Handle custom StringOrSlice type
	yaml.RegisterCustomUnmarshaler(func(s *StringOrSlice, data []byte) error {
		var str string
		if err := yaml.Unmarshal(data, &str); err == nil {
			*s = []string{str}
			return nil
		}
		var strSlice []string
		if err := yaml.Unmarshal(data, &strSlice); err == nil {
			*s = strSlice
			return nil
		}
		// Warn and ignore these type mismatches as metadata is not critical to conversion
		slog.Warn(fmt.Sprintf("type mismatch: expected string or []string: '%v' (%T)", data, data))
		return nil
	})
	err = yaml.Unmarshal(yamlData, metadata)
	return
}

func parseModelCardMetadata(fsys fs.FS, filepath string, metadata *ModelCardMetadata) (err error) {
	// Note: the underlying functions do record slog "warnings"
	yamlData, errParse := parseModelCardYAML(fsys, filepath)
	if errParse != nil {
		// Anticipating some model repos. have no HF YAML data, we will not return this first error
		return
	}
	errUnmarshal := unmarshalModelCardMetadata(yamlData, metadata)
	if errUnmarshal != nil {
		err = fmt.Errorf("model card metadata unmarshal failed (filepath: '%s'): %w", filepath, errUnmarshal)
	}
	return
}

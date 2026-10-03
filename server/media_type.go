package server

import (
	"fmt"

	"github.com/ollama/ollama/types/model"
)

type MediaTypeInfo struct {
	ShortName string
	Extension string
}

func ExportMediaTypeMap(name model.Name, config model.ConfigV2) map[string]MediaTypeInfo {
	modelExt := "." + config.ModelFormat
	modelName := fmt.Sprintf(
		"%s_%s_%s",
		name.Model,
		config.ModelType,
		config.FileType,
	)

	return map[string]MediaTypeInfo{
		"application/vnd.ollama.image.model":     {modelName, modelExt},
		"application/vnd.ollama.image.adapter":   {modelName + "_adapter", modelExt},
		"application/vnd.ollama.image.projector": {modelName + "_projector", modelExt},
		"application/vnd.ollama.image.tensor":    {modelName + "_tensor", modelExt},
		"application/vnd.ollama.image.draft":     {modelName + "_draft", modelExt},

		"application/vnd.ollama.image.params":   {"params", ""},
		"application/vnd.ollama.image.system":   {"system", ""},
		"application/vnd.ollama.image.json":     {"config", ".json"},
		"application/vnd.ollama.image.template": {"template", ""},
		"application/vnd.ollama.image.prompt":   {"prompt", ""},
		"application/vnd.ollama.image.license":  {"license", ""},

		"application/vnd.docker.container.image.v1+json": {"manifest", ".json"},
	}
}

var ImportMediaTypeMap = map[string]string{
	"license":       "application/vnd.ollama.image.license",
	"license.txt":   "application/vnd.ollama.image.license",
	"params":        "application/vnd.ollama.image.params",
	"system":        "application/vnd.ollama.image.system",
	"config":        "application/vnd.ollama.image.json",
	"config.json":   "application/vnd.ollama.image.json",
	"template":      "application/vnd.ollama.image.template",
	"template.tmpl": "application/vnd.ollama.image.template",
	"prompt":        "application/vnd.ollama.image.prompt",
	"prompt.tmpl":   "application/vnd.ollama.image.prompt",
}

package main

import (
	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
)

func modelOptions(model *server.Model, requestOpts map[string]interface{}) (api.Options, error) {
	opts := api.DefaultOptions()
	if err := opts.FromMap(model.Options); err != nil {
		return api.Options{}, err
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	return opts, nil
}

func sliceIntToInt64(slice []int) []int64 {
	result := make([]int64, len(slice))
	for i, v := range slice {
		result[i] = int64(v)
	}
	return result
}

func sliceInt64ToInt(slice []int64) []int {
	result := make([]int, len(slice))
	for i, v := range slice {
		result[i] = int(v)
	}
	return result
}

func convertToImageData(byteSlices [][]byte) []api.ImageData {
	result := make([]api.ImageData, len(byteSlices))
	for i, byteSlice := range byteSlices {
		result[i] = api.ImageData(byteSlice)
	}
	return result
}

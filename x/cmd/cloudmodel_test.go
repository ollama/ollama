package cmd

import (
	"errors"
	"testing"
)

func TestCloudModelSwitchRequest(t *testing.T) {
	// Test the error type
	req := &CloudModelSwitchRequest{Model: "glm-4.7:cloud"}

	// Test Error() method
	errMsg := req.Error()
	expected := "switch to model: glm-4.7:cloud"
	if errMsg != expected {
		t.Errorf("expected %q, got %q", expected, errMsg)
	}

	// Test errors.As
	var err error = req
	var switchReq *CloudModelSwitchRequest
	if !errors.As(err, &switchReq) {
		t.Error("errors.As should return true for CloudModelSwitchRequest")
	}

	if switchReq.Model != "glm-4.7:cloud" {
		t.Errorf("expected model glm-4.7:cloud, got %s", switchReq.Model)
	}
}

func TestSuggestedCloudModels(t *testing.T) {
	// Verify the suggested models are defined
	if len(suggestedCloudModels) == 0 {
		t.Error("suggestedCloudModels should not be empty")
	}

	// Check first model
	if suggestedCloudModels[0].Name != "glm-4.7:cloud" {
		t.Errorf("expected first model to be glm-4.7:cloud, got %s", suggestedCloudModels[0].Name)
	}
}

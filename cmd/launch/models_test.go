package launch

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSortModelList(t *testing.T) {
	models := []ModelItem{
		{Name: "Recommended1"},
		{Name: "Recommended2"},
		{Name: "Model1"},
		{Name: "Model2"},
		{Name: "Model3"},
		{Name: "Model4"},
		{Name: "Model5"},
	}
	recommendations := []ModelItem{
		{Name: "Recommended2"},
		{Name: "Recommended1"},
	}
	current := "Model2"
	checked := map[string]bool{"Model2": true, "Model3": true}
	notInstalled := map[string]bool{"Model4": true}

	sortModelList(models, recommendations, current, checked, notInstalled)

	expectedOrder := []string{
		"Recommended2", // Recommended, highest rank
		"Recommended1", // Recommended
		"Model2",       // Checked, current
		"Model3",       // Checked
		"Model1",       // Installed, alphabetical
		"Model5",       // Installed, alphabetical
		"Model4",       // Not installed
	}

	assert.Equal(t, expectedOrder, mapModelNames(models))
}

func mapModelNames(models []ModelItem) []string {
	names := []string{}
	for _, m := range models {
		names = append(names, m.Name)
	}

	return names
}

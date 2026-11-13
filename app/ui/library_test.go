//go:build windows || darwin

package ui

import (
	"strings"
	"testing"
)

// libraryFixture mirrors the markup of two rows from ollama.com/library: one
// with capabilities, variants, pulls and tags, and one with only a description.
const libraryFixture = `
<div id="repo">
  <ul role="list" class="grid grid-cols-1 gap-y-3">
    <li  class="flex items-baseline border-b border-neutral-200 py-6">
      <a href="/library/llama3.1" class="group w-full space-y-5">
        <div  title="llama3.1" class="flex flex-col">
          <h2 class="truncate text-xl font-medium underline-offset-2 md:text-2xl">
            <div class="flex space-x-2 items-center">
              <span class="group-hover:underline truncate">llama3.1</span>
            </div>
          </h2>
          <p class="max-w-lg break-words text-neutral-800 text-md">Llama 3.1 is a new state-of-the-art model from Meta available in 8B, 70B and 405B parameter sizes.</p>
        </div>
        <div class="flex flex-col space-y-2">
          <div class="flex flex-wrap space-x-2">
              <span  class="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-600 sm:text-[13px]">tools</span>
              <span  class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2 py-0.5 text-xs font-medium text-blue-600 sm:text-[13px]">8b</span>
              <span  class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2 py-0.5 text-xs font-medium text-blue-600 sm:text-[13px]">70b</span>
              <span  class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2 py-0.5 text-xs font-medium text-blue-600 sm:text-[13px]">405b</span>
          </div>
          <p class="flex space-x-5 text-[13px] font-medium text-neutral-500">
              <span class="flex items-center">
                <span >121.1M</span>
                <span class="hidden sm:flex">&nbsp;Pulls</span>
              </span>
              <span class="flex items-center">
                <span >93</span>
                <span class="hidden sm:flex">&nbsp;Tags</span>
              </span>
          </p>
        </div>
      </a>
    </li>
    <li  class="flex items-baseline border-b border-neutral-200 py-6">
      <a href="/library/nomic-embed-text" class="group w-full space-y-5">
        <div  title="nomic-embed-text" class="flex flex-col">
          <h2 class="truncate text-xl font-medium underline-offset-2 md:text-2xl">
            <div class="flex space-x-2 items-center">
              <span class="group-hover:underline truncate">nomic-embed-text</span>
            </div>
          </h2>
          <p class="max-w-lg break-words text-neutral-800 text-md">A high-performing open embedding model with a large token context window.</p>
        </div>
        <div class="flex flex-col space-y-2">
          <div class="flex flex-wrap space-x-2">
              <span  class="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-600 sm:text-[13px]">embedding</span>
          </div>
          <p class="flex space-x-5 text-[13px] font-medium text-neutral-500">
              <span class="flex items-center">
                <span >97.7M</span>
                <span class="hidden sm:flex">&nbsp;Pulls</span>
              </span>
              <span class="flex items-center">
                <span >3</span>
                <span class="hidden sm:flex">&nbsp;Tags</span>
              </span>
          </p>
        </div>
      </a>
    </li>
  </ul>
</div>
`

func TestParseLibrary(t *testing.T) {
	models, err := parseLibrary(strings.NewReader(libraryFixture))
	if err != nil {
		t.Fatalf("parseLibrary() error = %v", err)
	}

	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d: %+v", len(models), models)
	}

	llama := models[0]
	if llama.Model != "llama3.1" {
		t.Fatalf("expected model llama3.1, got %q", llama.Model)
	}
	if !strings.Contains(llama.Description, "state-of-the-art model from Meta") {
		t.Fatalf("unexpected description: %q", llama.Description)
	}
	if len(llama.Capabilities) != 1 || llama.Capabilities[0] != "tools" {
		t.Fatalf("unexpected capabilities: %+v", llama.Capabilities)
	}
	if len(llama.Variants) != 3 || llama.Variants[0] != "8b" || llama.Variants[2] != "405b" {
		t.Fatalf("unexpected variants: %+v", llama.Variants)
	}
	if llama.Pulls != "121.1M" {
		t.Fatalf("unexpected pulls: %q", llama.Pulls)
	}
	if llama.Tags != 93 {
		t.Fatalf("unexpected tags: %d", llama.Tags)
	}

	embed := models[1]
	if embed.Model != "nomic-embed-text" {
		t.Fatalf("expected model nomic-embed-text, got %q", embed.Model)
	}
	if len(embed.Capabilities) != 1 || embed.Capabilities[0] != "embedding" {
		t.Fatalf("unexpected capabilities: %+v", embed.Capabilities)
	}
	if len(embed.Variants) != 0 {
		t.Fatalf("expected no variants, got %+v", embed.Variants)
	}
}

func TestParseLibraryUnescapesDescriptions(t *testing.T) {
	fixture := `<li  class="flex items-baseline border-b border-neutral-200 py-6">
      <a href="/library/llama3.2" class="group w-full space-y-5">
          <p class="max-w-lg break-words text-neutral-800 text-md">Meta&#39;s Llama 3.2 goes small with 1B and 3B models.</p>
      </a>
    </li>`

	models, err := parseLibrary(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("parseLibrary() error = %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].Description != "Meta's Llama 3.2 goes small with 1B and 3B models." {
		t.Fatalf("expected HTML entities to be unescaped, got %q", models[0].Description)
	}
}

func TestParseLibrarySkipsRowsWithoutModelName(t *testing.T) {
	fixture := `<li  class="flex items-baseline border-b border-neutral-200 py-6">
      <a href="/not-a-library-link" class="group w-full space-y-5">
          <p class="max-w-lg break-words text-neutral-800 text-md">No model name here.</p>
      </a>
    </li>`

	models, err := parseLibrary(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("parseLibrary() error = %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("expected rows without a model name to be skipped, got %+v", models)
	}
}

func TestParseLibraryEmptyPage(t *testing.T) {
	models, err := parseLibrary(strings.NewReader("<html><body>no models</body></html>"))
	if err != nil {
		t.Fatalf("parseLibrary() error = %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("expected no models, got %+v", models)
	}
}

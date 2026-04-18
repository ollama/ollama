package server

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"github.com/ollama/ollama/api"
)

const (
	DefaultSearchUrl = "https://registry.ollama.ai/search"
)

func flaky(selection *goquery.Selection) string {
	if selection != nil {
		return strings.Trim(strings.Trim(selection.Text(), "\n"), " ")
	}
	return ""
}

func parseSearchResults(resp *http.Response) (*api.SearchResponse, error) {
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	results := &api.SearchResponse{}

	// main parsing logic
	// all search results are in this div
	doc.Find("#searchresults > ul > li").Each(func(i int, s *goquery.Selection) {
		desc := flaky(s.Find("p").First())
		modelName := flaky(s.Find("span[x-test-search-response-title]"))
		capability := flaky(s.Find("span[x-test-capability]"))

		var modelSizes []string
		s.Find("span[x-test-size]").Each(func(j int, span *goquery.Selection) {
			modelSizes = append(modelSizes, flaky(span.First()))
		})

		pullCount := flaky(s.Find("span[x-test-pull-count]"))

		results.Models = append(results.Models, api.SearchModelResponse{
			Name:        modelName,
			Description: desc,
			Category:    capability,
			Sizes:       strings.Join(modelSizes, ","),
			Pulls:       pullCount,
		})
	})

	return results, nil
}

func formatSearchRequestURL(params *api.SearchRequest) (*url.URL, error) {
	// base URL
	u, err := url.Parse(DefaultSearchUrl)
	if err != nil {
		return nil, err
	}

	// query path parameters
	query := u.Query()
	query.Add("c", params.Category)
	query.Add("o", params.Order)
	query.Add("q", params.Query)
	u.RawQuery = query.Encode()
	return u, nil
}

func SearchRegistry(ctx context.Context, params *api.SearchRequest) (*api.SearchResponse, error) {
	requestURL, err := formatSearchRequestURL(params)
	if err != nil {
		return nil, err
	}
	headers := make(http.Header)

	resp, err := makeRequest(ctx, http.MethodGet, requestURL, headers, nil, &registryOptions{})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		// we messed up somewhere :(
		return nil, errors.New("error - can't search default registry at " + DefaultSearchUrl)
	}

	return parseSearchResults(resp)
}

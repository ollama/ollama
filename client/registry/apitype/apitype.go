package apitype

import (
	"cmp"
	"encoding/json"
	"log/slog"
	"net/url"
	"slices"
)

type Manifest struct {
	Layers []*Layer `json:"layers"`
}

type CompletePart struct {
	URL  string `json:"url"` // contains partNumber and uploadId from server
	ETag string `json:"etag"`
}

func queryFromString(s string) url.Values {
	u, err := url.Parse(s)
	if err != nil {
		return nil
	}
	return u.Query()
}

func (cp *CompletePart) Compare(o *CompletePart) int {
	qa := queryFromString(cp.URL)
	qb := queryFromString(o.URL)
	return cmp.Or(
		cmp.Compare(qa.Get("partNumber"), qb.Get("partNumber")),
		cmp.Compare(qa.Get("uploadId"), qb.Get("uploadId")),
		cmp.Compare(cp.ETag, o.ETag),
	)
}

func SortCompleteParts(a []*CompletePart) {
	slices.SortFunc(a, (*CompletePart).Compare)
}

type Layer struct {
	Digest    string `json:"digest"`
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`

	// If present, URL is a remote location of the layer for fetching.
	URL string `json:"url,omitempty"`
}

func (l *Layer) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("digest", l.Digest),
		slog.String("mediaType", l.MediaType),
		slog.Int64("size", l.Size),
		slog.String("url", l.URL),
	)
}

type PushRequest struct {
	Name     string          `json:"ref"`
	Manifest json.RawMessage `json:"manifest,omitempty"`

	// Parts is a list of upload parts that the client upload in the previous
	// push.
	CompleteParts []*CompletePart `json:"part_uploads"`
}

type Need struct {
	Digest string `json:"digest"`

	Start int64 `json:"start"`
	End   int64 `json:"end"`

	// URL is the url to PUT the layer to.
	//
	// Clients must include it as the URL, along with the ETag in the
	// response headers from the PUT request, in the next push request
	// in the Uploaded field.
	URL string `json:"url"`
}

type PushResponse struct {
	// Needs is a list of digests that the client needs to push before
	// repushing the manifest.
	Needs []*Need `json:"requirements,omitempty"`
}

type PullResponse struct {
	// Name is the name of the model being pulled.
	Name string `json:"name"`

	// Manifest is the manifest of the model being pulled.
	Manifest *Manifest `json:"manifest"`
}

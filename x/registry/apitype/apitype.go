package apitype

import "encoding/json"

type Manifest struct {
	Layers []Layer `json:"layers"`
}

type CompletePart struct {
	URL  string `json:"url"` // contains partNumber and uploadId from server
	ETag string `json:"etag"`
}

type Layer struct {
	Digest    string `json:"digest"`
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
}

type PushRequest struct {
	Name     string          `json:"ref"`
	Manifest json.RawMessage `json:"manifest"`

	// Parts is a list of upload parts that the client upload in the previous
	// push.
	CompleteParts []CompletePart `json:"part_uploads"`
}

type Requirement struct {
	Digest string `json:"digest"`
	Offset int64  `json:"offset"`
	Size   int64  `json:"Size"`

	// URL is the url to PUT the layer to.
	//
	// Clients must include it as the URL,  alond with the ETag in the
	// response headers from the PUT request, in the next push request
	// in the Uploaded field.
	URL string `json:"url"`
}

type PushResponse struct {
	// Requirements is a list of digests that the client needs to push before
	// repushing the manifest.
	Requirements []Requirement `json:"requirements,omitempty"`
}

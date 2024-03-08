package registry

type Manifest struct {
	Layers []Layer `json:"layers"`
}

type Layer struct {
	Digest    string `json:"digest"`
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
}

type PushRequest struct {
	Manifest Manifest `json:"manifest"`
}

type Requirement struct {
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
	URL    string `json:"url"`
}

type PushResponse struct {
	// Requirements is a list of digests that the client needs to push before
	// repushing the manifest.
	Requirements []Requirement `json:"requirements,omitempty"`
}

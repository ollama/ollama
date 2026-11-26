package model

// StructuredOutputFrames describes the markers a model can emit before a
// structured answer. A thinking close without an open continues a thought
// the template already opened.
type StructuredOutputFrames struct {
	ThinkingOpen     string
	ThinkingClose    string
	ThinkingExcludes []string
	ContentOpen      string
}

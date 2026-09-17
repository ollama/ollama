package renderers

type OrnithRenderer struct {
	Qwen35Renderer
}

func newOrnithRenderer() Renderer {
	return &OrnithRenderer{
		Qwen35Renderer: Qwen35Renderer{
			isThinking:                      true,
			alwaysRenderAssistantThinkBlock: true,
			emitEmptyThinkOnNoThink:         true,
			useImgTags:                      RenderImgTags,
		},
	}
}

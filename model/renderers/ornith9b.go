package renderers

type Ornith9BRenderer struct {
	Qwen35Renderer
}

func newOrnith9BRenderer() Renderer {
	return &Ornith9BRenderer{
		Qwen35Renderer: Qwen35Renderer{
			isThinking:                      true,
			alwaysRenderAssistantThinkBlock: true,
			emitEmptyThinkOnNoThink:         true,
			useImgTags:                      RenderImgTags,
		},
	}
}

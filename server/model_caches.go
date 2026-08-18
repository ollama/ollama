package server

import "context"

type modelCaches struct {
	recommendations *modelRecommendationsCache
	show            *modelShowCache
}

func newModelCaches() *modelCaches {
	return &modelCaches{
		recommendations: newModelRecommendationsCache(),
		show:            newModelShowCache(),
	}
}

func (c *modelCaches) Start(ctx context.Context) {
	if c == nil {
		return
	}
	if c.recommendations != nil {
		c.recommendations.Start(ctx)
	}
	if c.show != nil {
		c.show.Start(ctx)
	}
}

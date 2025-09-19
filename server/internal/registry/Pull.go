func (r *Registry) Pull(ctx context.Context, name string) error {
	var expected int64
	for _, l := range layers {
		expected += l.Size
	}

	var received atomic.Int64
	var g errgroup.Group
	g.SetLimit(r.maxStreams())

	for _, layer := range layers {
		if isLayerCached(layer, c, &received, t) {
			continue
		}

		chunked, err := c.Chunked(layer.Digest, layer.Size)
		if err != nil {
			t.update(layer, 0, err)
			continue
		}

		wg := &sync.WaitGroup{}
		if err := downloadLayerChunks(ctx, name, layer, chunked, r, g, wg, t, &received); err != nil {
			t.update(layer, 0, err)
			continue
		}

		// Fechar o chunked writer após todos os chunks terminarem
		g.Go(func() error {
			wg.Wait()
			chunked.Close()
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}
	if received.Load() != expected {
		return fmt.Errorf("%w: received %d/%d", ErrIncomplete, received.Load(), expected)
	}

	return nil
}

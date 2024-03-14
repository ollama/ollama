package main

import (
	"context"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
	"github.com/taubyte/vm-orbit/satellite"
)

type pull struct {
	created time.Time
	status  api.ProgressResponse
	err     error
}

func (s *ollama) getPullId(model string) (uint64, func(api.ProgressResponse)) {
	id := hash(model)
	s.pullLock.Lock()
	defer s.pullLock.Unlock()
	if _, ok := s.pulls[id]; ok {
		return id, nil
	} else {
		s.pulls[id] = &pull{
			created: time.Now(),
		}
		return id, func(pr api.ProgressResponse) {
			s.pullLock.Lock()
			defer s.pullLock.Unlock()
			s.pulls[id].status = pr
		}
	}
}

func (s *ollama) W_pull(
	ctx context.Context,
	module satellite.Module,
	modelNamePtr uint32,
	modelNameSize uint32,
	pullIdptr uint32,
) Error {
	model, err := module.ReadString(modelNamePtr, modelNameSize)
	if err != nil {
		return ErrorReadMemory
	}

	id, updateFunc := s.getPullId(model)

	if updateFunc != nil {
		go func() {
			err = server.PullModel(s.ctx, model, &server.RegistryOptions{}, updateFunc)
			s.pullLock.Lock()
			defer s.pullLock.Unlock()
			s.pulls[id].err = err
			// TODO: add crean up mechanism
		}()
	}

	module.WriteUint64(pullIdptr, id)

	return ErrorNone
}

func (s *ollama) W_pull_status(
	ctx context.Context,
	module satellite.Module,
	pullId uint64,
	statusBufferPtr uint32,
	statusBufferSize uint32,
	statusBufferWrittenPtr uint32,
	totalPtr uint32,
	completedPtr uint32,
	errorBufferPtr uint32,
	errorBufferSize uint32,
	errorBufferWrittenPtr uint32,
) Error {
	s.pullLock.RLock()
	defer s.pullLock.RUnlock()
	p, ok := s.pulls[pullId]
	if !ok {
		return ErrorPullNotFound
	}

	if int(statusBufferSize) >= len(p.status.Status) {
		statusBufferWritten, _ := module.WriteString(statusBufferPtr, p.status.Status)
		module.WriteUint32(statusBufferWrittenPtr, statusBufferWritten)
	}

	if p.err != nil && int(errorBufferSize) >= len(p.err.Error()) {
		errorBufferWritten, _ := module.WriteString(errorBufferPtr, p.err.Error())
		module.WriteUint32(errorBufferWrittenPtr, errorBufferWritten)
	}

	module.WriteUint64(totalPtr, uint64(p.status.Total))
	module.WriteUint64(completedPtr, uint64(p.status.Completed))

	return ErrorNone
}

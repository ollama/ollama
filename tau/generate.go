package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"strings"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/server"
	sdk "github.com/samyfodil/go-ollama-tau-sdk"
	"github.com/taubyte/vm-orbit/satellite"

	"github.com/jmorganca/ollama/llm"

	"github.com/fxamacker/cbor/v2"
)

type genJob struct {
	ch   chan api.GenerateResponse
	err  error
	ctx  context.Context
	ctxC context.CancelFunc
}

func (s *ollama) getGenId(ctx context.Context, prompt string) (uint64, *genJob, error) {
	s.generateLock.Lock()
	defer s.generateLock.Unlock()

	var id uint64
genId:
	for {
		select {
		case <-ctx.Done():
			return 0, nil, errors.New("failed to generate id with context expired")
		default:
			id = hash(fmt.Sprintf("%s|%d", prompt, time.Now().UnixNano()))
			if _, ok := s.generateJobs[id]; !ok {
				break genId
			}
		}
	}

	job := &genJob{
		ch: make(chan api.GenerateResponse, 256),
	}

	job.ctx, job.ctxC = context.WithCancel(s.ctx)

	s.generateJobs[id] = job

	return id, job, nil
}

func (s *ollama) W_generate(
	ctx context.Context,
	module satellite.Module,

	modelNamePtr uint32,
	modelNameSize uint32,

	promptPtr uint32,
	promptSize uint32,

	systemPtr uint32,
	systemSize uint32,

	templatePtr uint32,
	templateSize uint32,

	contextPtr uint32, // []int64
	contextSize uint32,

	rawVal uint32, // 0=false or 1=true

	keepaliveDur uint64,

	imagesPtr uint32,
	imagesSize uint32,

	optionsPtr uint32, // cbor
	optionsSize uint32,

	errBufferPtr uint32,
	errBufferSize uint32,
	errBufferWrittenPtr uint32,

	idPtr uint32,
) Error {

	checkpointStart := time.Now()

	modelName, err := module.ReadString(modelNamePtr, modelNameSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "model name")
	}

	prompt, err := module.ReadString(promptPtr, promptSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "prompt")
	}

	system, err := module.ReadString(systemPtr, systemSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "system")
	}

	template, err := module.ReadString(templatePtr, templateSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "template")
	}

	contextPayload, err := module.MemoryRead(contextPtr, contextSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "context")
	}

	_context, err := sdk.BytesToInt64Slice(contextPayload)
	if err != nil {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, fmt.Errorf("deconding context failed with %w", err))
	}

	context := sliceInt64ToInt(_context)

	raw := false
	if rawVal == 1 {
		raw = true
	}

	keepalive := time.Duration(keepaliveDur)
	if keepalive == 0 {
		keepalive = defaultSessionDuration
	}

	imagesPlayload, err := module.MemoryRead(imagesPtr, imagesSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "images")
	}

	bimages, err := sdk.BytesToBytesSlice(imagesPlayload)
	if err != nil {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, fmt.Errorf("deconding images failed with %w", err))
	}

	images := convertToImageData(bimages)

	optionsPayload, err := module.MemoryRead(optionsPtr, optionsSize)
	if err != nil {
		return returnMemoryRead(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, "options")
	}

	var _options map[string]interface{}

	if len(optionsPayload) > 0 {
		err = cbor.Unmarshal(optionsPayload, &_options)
		if err != nil {
			return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, fmt.Errorf("parsing options failed with %w", err))
		}
	}

	if prompt == "" && template == "" && system == "" {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, errors.New("Empty"))
	}

	model, err := server.GetModel(modelName)
	if err != nil {
		var pErr *fs.PathError
		if errors.As(err, &pErr) {
			return ErrorModelNotFound
		}
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
	}

	if model.IsEmbedding() {
		return ErrorEmbeddingNotSupportedInGenerate
	}

	options, err := modelOptions(model, _options)
	if err != nil {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
	}

	if err := s.load(model, options, keepalive); err != nil {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
	}

	checkpointLoaded := time.Now()

	switch {
	case raw:
	case prompt != "":
		if template == "" {
			template = model.Template
		}

		if system == "" {
			system = model.System
		}

		var sb strings.Builder
		if context != nil {
			prev, err := loaded.runner.Decode(ctx, context)
			if err != nil {
				return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
			}

			sb.WriteString(prev)
		}

		// write image tags
		// TODO: limit the number of images to fit in the context similar to the chat endpoint
		for i := range images {
			prompt += fmt.Sprintf(" [img-%d]", i)
		}

		p, err := server.Prompt(template, system, prompt, "", true)
		if err != nil {
			return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
		}

		sb.WriteString(p)

		prompt = sb.String()
	}

	jobId, job, err := s.getGenId(ctx, prompt)
	if err != nil {
		return returnError(module, errBufferPtr, errBufferSize, errBufferWrittenPtr, err)
	}

	slog.Debug("generate handler", "job", jobId)

	var generated strings.Builder
	go func() {
		defer close(job.ch)

		fn := func(r llm.PredictResult) {
			// Update model expiration
			loaded.expireAt = time.Now().Add(keepalive)
			loaded.expireTimer.Reset(keepalive)

			// Build up the full response
			if _, err := generated.WriteString(r.Content); err != nil {
				job.err = err
				return
			}

			resp := api.GenerateResponse{
				Model:     modelName,
				CreatedAt: time.Now().UTC(),
				Done:      r.Done,
				Response:  r.Content,
				Metrics: api.Metrics{
					PromptEvalCount:    r.PromptEvalCount,
					PromptEvalDuration: r.PromptEvalDuration,
					EvalCount:          r.EvalCount,
					EvalDuration:       r.EvalDuration,
				},
			}

			if r.Done {
				resp.TotalDuration = time.Since(checkpointStart)
				resp.LoadDuration = checkpointLoaded.Sub(checkpointStart)

				if !raw {
					p, err := server.Prompt(template, system, prompt, generated.String(), false)
					if err != nil {
						job.err = err
						return
					}

					// TODO (jmorganca): encode() should not strip special tokens
					tokens, err := loaded.runner.Encode(job.ctx, p)
					if err != nil {
						job.err = err
						return
					}

					resp.Context = append(context, tokens...)
				}
			}

			job.ch <- resp
		}

		var imagesData []llm.ImageData
		for i := range images {
			imagesData = append(imagesData, llm.ImageData{
				ID:   i,
				Data: images[i],
			})
		}

		// Start prediction
		predictReq := llm.PredictOpts{
			Prompt:  prompt,
			Format:  "",
			Images:  imagesData,
			Options: options,
		}
		if err := loaded.runner.Predict(job.ctx, predictReq, fn); err != nil {
			job.err = err
		}
	}()

	module.WriteUint64(idPtr, jobId)

	return ErrorNone
}

var tokenDefaultWait = time.Second

func (s *ollama) W_next(
	ctx context.Context,
	module satellite.Module,

	jobId uint64,

	wait uint64, // default is TokenDefaultWait

	tokenBufferPtr uint32,
	tokenBufferSize uint32,
	tokenBufferWrittenPtr uint32,

	errBufferPtr uint32,
	errBufferSize uint32,
	errBufferWrittenPtr uint32,

) Error {
	s.generateLock.RLock()
	job, exists := s.generateJobs[jobId]
	s.generateLock.RUnlock()

	if !exists {
		return ErrorJobNotFound
	}

	timeout := time.Duration(wait)
	if timeout == 0 {
		timeout = tokenDefaultWait
	}

	select {
	case token, ok := <-job.ch:
		if !ok {
			return ErrorEOF
		}

		t := []byte(token.Response)
		if uint32(len(t)) > tokenBufferSize {
			return ErrorBufferTooSmall
		}
		n, err := module.MemoryWrite(tokenBufferPtr, t)
		if err != nil {
			return ErrorWriteMemory
		}
		module.WriteUint32(tokenBufferWrittenPtr, n)

		return ErrorNone
	case <-time.After(timeout):
		return ErrorTimeout
	case <-ctx.Done():
		return ErrorTimeout
	}
}

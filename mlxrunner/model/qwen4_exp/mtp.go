package qwen4_exp

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type MTPHead struct {
	EmbeddingNorm *nn.RMSNorm
	HiddenNorm    *nn.RMSNorm
	FCEmbedding   nn.LinearLayer
	FCHidden      nn.LinearLayer
	Mixer         *hyperConnection
	Layer         *Layer
}

func (m *Model) loadMTP(linears model.LinearFactory, tensors map[string]*mlx.Array) (*MTPHead, error) {
	head := &MTPHead{}
	var err error
	if head.EmbeddingNorm, err = loadNorm(tensors, "mtp.pre_fc_norm_embedding", m.RMSNormEps); err != nil {
		return nil, err
	}
	if head.HiddenNorm, err = loadNorm(tensors, "mtp.pre_fc_norm_hidden", m.RMSNormEps); err != nil {
		return nil, err
	}
	if head.FCEmbedding, err = requiredLinear(linears, "mtp.fc_embedding"); err != nil {
		return nil, err
	}
	if head.FCHidden, err = requiredLinear(linears, "mtp.fc_hidden"); err != nil {
		return nil, err
	}
	if head.Mixer, err = loadHyperConnection(linears, tensors, "mtp.hyper_connection_mixer", false, m.Config); err != nil {
		return nil, err
	}
	if head.Layer, err = m.loadLayer(linears, tensors, "mtp.layers.0", m.Config.MTP.LayerTypes[0], -1); err != nil {
		return nil, err
	}
	return head, nil
}

// mtpDraft views the checkpoint's inline predictor through DraftModel.
type mtpDraft Model

func (m *Model) SelfDraft() model.DraftModel {
	if m.MTP == nil {
		return nil
	}
	return (*mtpDraft)(m)
}

func (m *mtpDraft) NewCaches() []cache.Cache {
	if m.MTP == nil {
		return nil
	}
	return []cache.Cache{cache.NewKVCache(), cache.NewKVCache()}
}

func (m *mtpDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (m *mtpDraft) Unembed(x *mlx.Array) *mlx.Array { return (*Model)(m).Unembed(x) }

func (m *mtpDraft) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	B, L := int32(b.InputIDs.Dim(0)), int32(b.InputIDs.Dim(1))
	var mrope []int32
	if m.Vision != nil {
		mrope = m.Vision.RopePositions(b, L)
	}
	ropePositions := canonicalRopePositionRows(b, L, mrope)

	embedding := m.EmbedTokens.Forward(b.InputIDs)
	// The reference MTP is text-only: visual information reaches it through
	// the target's multi-stream hidden state, while this branch embeds raw IDs.
	embedding = m.MTP.FCEmbedding.Forward(m.MTP.EmbeddingNorm.Forward(embedding, m.RMSNormEps))

	backbone := m.MTP.HiddenNorm.Forward(b.Hidden, m.RMSNormEps)
	backbone = mlx.Reshape(backbone, B, L, m.HCCount, m.HiddenSize)
	backbone = m.MTP.FCHidden.Forward(backbone)
	backbone = mlx.Add(backbone, mlx.ExpandDims(embedding, -2))
	backbone = mlx.Reshape(backbone, B, L, m.HCCount*m.HiddenSize)

	var main, side cache.Cache
	if len(draftCaches) > 0 {
		main = draftCaches[0]
	}
	if len(draftCaches) > 1 {
		side = draftCaches[1]
	}
	multi := m.MTP.Layer.Forward(backbone, b, main, side, ropePositions, m.Config)
	sample := m.MTP.Mixer.Reduce(multi, m.Config)
	return sample, multi
}

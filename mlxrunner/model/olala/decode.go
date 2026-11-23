package olala

import (
	"log/slog"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// fastLayer holds one layer's weights for the fused decode path (see
// mlx/olala_kernels.go). Fields are exported so that mlx.Collect keeps them.
type fastLayer struct {
	In, InSmall, MixerProj *mlx.OlalaWeight
	Router, InCat, OutCat  *mlx.OlalaWeight
	Up, Down               *mlx.OlalaWeight
	// M3Prm packs the Mamba-3 per-head parameters (M layers).
	M3Prm *mlx.Array
	// QNorm, KNorm, Scaler are the attention parameters (V layers).
	QNorm, KNorm, Scaler *mlx.Array
	// GeoMixer, GeoMlp are [scale, bias, layer + 1] of the geodesic updates.
	GeoMixer, GeoMlp *mlx.Array
	EBias            *mlx.Array
}

func olalaWeight(l nn.LinearLayer) *mlx.OlalaWeight {
	switch l := l.(type) {
	case *nn.Linear:
		if l.Bias != nil {
			return nil
		}
		return mlx.NewOlalaWeight(l.Weight, nil, nil, 0, 0)
	case *nn.QuantizedLinear:
		if l.Mode != "affine" || l.Bias != nil || l.GlobalScale != nil {
			return nil
		}
		return mlx.NewOlalaWeight(l.Weight, l.Scales, l.QBiases, l.Bits, l.GroupSize)
	}
	return nil
}

func expertWeight(e expertWeights) *mlx.OlalaWeight {
	if e.Quantized {
		if e.Mode != "affine" {
			return nil
		}
		return mlx.NewOlalaWeight(e.Raw, e.Scales, e.Biases, e.Bits, e.GroupSize)
	}
	return mlx.NewOlalaWeight(e.Raw, nil, nil, 0, 0)
}

func geoParams(scale, bias *mlx.Array, layer int) *mlx.Array {
	return mlx.Concatenate([]*mlx.Array{
		mlx.Reshape(scale, 1),
		mlx.Reshape(bias, 1),
		mlx.FromValues([]float32{float32(layer + 1)}, 1),
	}, 0)
}

// buildFast prepares the fused decode path, or returns nil when a weight is
// in a format the kernels do not read.
func (m *Model) buildFast() []*fastLayer {
	if !mlx.OlalaKernelsAvailable() {
		return nil
	}
	cfg := m.cfg
	if cfg.HeadDim != 128 || cfg.mN != 128 || cfg.mA != 32 || cfg.NumAttentionHeads != 4*cfg.nNoise {
		return nil
	}
	flat := func(a *mlx.Array) *mlx.Array { return mlx.Reshape(f32(a), -1) }
	layers := make([]*fastLayer, len(m.Layers))
	var eval []*mlx.Array
	for i, l := range m.Layers {
		f := &fastLayer{
			MixerProj: olalaWeight(l.MixerProj),
			Router:    mlx.NewOlalaWeight(l.Moe.GateW, nil, nil, 0, 0),
			InCat:     olalaWeight(l.Moe.InCat),
			OutCat:    olalaWeight(l.Moe.OutCat),
			Up:        expertWeight(l.Moe.Up),
			Down:      expertWeight(l.Moe.Down),
			GeoMixer:  geoParams(l.GeoMixS, l.GeoMixB, i),
			GeoMlp:    geoParams(l.GeoMlpS, l.GeoMlpB, i),
			EBias:     l.Moe.ExpertBias,
		}
		ws := []*mlx.OlalaWeight{f.MixerProj, f.Router, f.InCat, f.OutCat, f.Up, f.Down}
		if l.typ == 'M' {
			mm := l.Mamba
			f.In = olalaWeight(mm.InProj)
			f.M3Prm = mlx.Concatenate([]*mlx.Array{
				flat(mm.DtBias), flat(mm.D), flat(mm.WB), flat(mm.WC), flat(mm.BBias), flat(mm.CBias),
				flat(mm.MimoX), flat(mm.MimoZ), flat(mm.OutMim),
			}, 0)
			ws = append(ws, f.In)
			eval = append(eval, f.M3Prm)
		} else {
			a := l.Attn
			f.In = olalaWeight(a.InProj)
			f.InSmall = olalaWeight(a.InProjSmall)
			f.QNorm, f.KNorm, f.Scaler = a.QNorm, a.KNorm, a.Scaler
			ws = append(ws, f.In, f.InSmall)
		}
		for _, w := range ws {
			if w == nil {
				slog.Info("olala: fused decode disabled, unsupported weight format", "layer", i)
				return nil
			}
			eval = append(eval, w.W)
		}
		eval = append(eval, f.GeoMixer, f.GeoMlp)
		layers[i] = f
	}
	mlx.Eval(eval...)
	for _, f := range layers {
		for _, w := range []*mlx.OlalaWeight{f.In, f.InSmall, f.MixerProj, f.Router, f.InCat, f.OutCat, f.Up, f.Down} {
			if w != nil {
				w.Eval()
			}
		}
	}
	return layers
}

// canDecodeFused reports whether this forward can take the fused decode path:
// one token of one sequence, with the model's own caches.
func (m *Model) canDecodeFused(b *batch.Batch, caches []cache.Layer) bool {
	if m.Fast == nil || len(caches) < len(m.Layers) {
		return false
	}
	dims := b.InputIDs.Dims()
	if dims[0] != 1 || dims[1] != 1 || len(b.SeqOffsets) != 1 {
		return false
	}
	for i, l := range m.Layers {
		if l.typ == 'M' {
			if _, ok := caches[i].(cache.Recurrent); !ok {
				return false
			}
			continue
		}
		if _, ok := caches[i].(*cache.KVCache); !ok {
			return false
		}
		if m.shiftSlot[i] >= len(caches) {
			return false
		}
		if _, ok := caches[m.shiftSlot[i]].(cache.Recurrent); !ok {
			return false
		}
	}
	return true
}

// decodeFused runs one decode step through the fused kernels and returns the
// final residual (1, 1, D) f32.
func (m *Model) decodeFused(b *batch.Batch, caches []cache.Layer) *mlx.Array {
	cfg := m.cfg
	D := int(cfg.HiddenSize)
	const T = 1
	m3 := mlx.OlalaM3Dims{H: int(cfg.mH), P: int(cfg.mP), N: int(cfg.mN), R: int(cfg.mR), A: int(cfg.mA), NStatic: int(cfg.nStatic)}
	att := mlx.OlalaAttnDims{
		Hq: int(cfg.NumAttentionHeads), Hk: int(cfg.nNoise), Dh: int(cfg.HeadDim), Rk: int(cfg.TPARank),
		NNoise: int(cfg.nNoise), Window: int(cfg.SlwWsize),
		Scale: cfg.attnScale, Softcap: cfg.SoftcapAttn, GateBias: cfg.gateBias,
	}
	kvw := att.Hk * att.Dh
	dLatent := int(cfg.RoutedInputDim)

	resid := mlx.Reshape(f32(m.Embed.Forward(b.InputIDs)), T, int32(D))
	pos := mlx.NewArrayInt32([]int32{b.SeqOffsets[0]}, []int32{1})
	var g, gst, prevGeo *mlx.Array

	for i, l := range m.Layers {
		f := m.Fast[i]
		ws := []*mlx.OlalaWeight{f.In}
		if l.typ == 'V' {
			ws = append(ws, f.InSmall)
		}
		var outs []*mlx.Array
		switch {
		case g == nil: // first layer: normalize the embedding
			outs = mlx.OlalaGEMV([]*mlx.Array{resid}, ws, T, D, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaNorm})
		case gst != nil:
			outs = mlx.OlalaGEMV([]*mlx.Array{resid, g, prevGeo, gst}, ws, T, D, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaGeos})
		default:
			outs = mlx.OlalaGEMV([]*mlx.Array{resid, g, prevGeo}, ws, T, D, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaGeo})
		}
		resid = outs[len(outs)-1]

		var ymix *mlx.Array
		if l.typ == 'M' {
			rc := caches[i].(cache.Recurrent)
			hist := rc.Get(b, mlx.DTypeFloat32)
			st := mlx.Reshape(hist.DeltaState(), T, int32(m3.H), int32(m3.StateWidth()))
			y, stOut := mlx.OlalaM3Step(outs[0], f.M3Prm, st, T, m3)
			rc.Put(b, []*mlx.Array{hist.ConvState()}, []*mlx.Array{mlx.Reshape(stOut, T, int32(m3.H), 1, int32(m3.StateWidth()))})
			ymix = y
		} else {
			big, small := outs[0], outs[1]
			sc := caches[m.shiftSlot[i]].(cache.Recurrent)
			shist := sc.Get(b, mlx.DTypeFloat32)
			conv := shist.ConvState() // (1, 1, 2*Hk*Dh)
			kprev := sliceLast(conv, 0, kvw)
			vprev := sliceLast(conv, kvw, 2*kvw)
			q, k, v, kraw, vraw := mlx.OlalaVPrep(big, small, kprev, vprev, f.QNorm, f.KNorm, f.Scaler, pos, T, att, m.computeDType())
			sc.Put(b, []*mlx.Array{mlx.Concatenate([]*mlx.Array{
				mlx.Reshape(kraw, T, 1, int32(kvw)), mlx.Reshape(vraw, T, 1, int32(kvw)),
			}, -1)}, []*mlx.Array{shist.DeltaState()})

			hist := caches[i].(*cache.KVCache).Update(b, k, v)
			keys, values := hist.K(), hist.V()
			S := keys.Dim(2)
			kstart := mlx.NewArrayInt32([]int32{int32(max(S-1-att.Window, 0))}, []int32{1})
			o, mm, ll := mlx.OlalaAttnDecode(q, keys, values, kstart, T, S, att)
			ymix = mlx.OlalaAttnCombine(o, mm, ll, small, big, T, att)
		}

		r := mlx.OlalaGEMV([]*mlx.Array{ymix}, []*mlx.OlalaWeight{f.MixerProj}, T, ymix.Dim(1), mlx.OlalaGEMVOptions{StatsX: resid})
		gmix, gmst := r[0], r[1]

		// MoE: geodesic(resid, gmix) -> router | in_cat -> routed experts -> out_cat
		r = mlx.OlalaGEMV([]*mlx.Array{resid, gmix, f.GeoMixer, gmst}, []*mlx.OlalaWeight{f.Router, f.InCat}, T, D,
			mlx.OlalaGEMVOptions{Prologue: mlx.OlalaGeos})
		logits, hcat := r[0], r[1]
		resid = r[2]
		parts := mlx.OlalaMoE(logits, f.EBias, hcat, hcat.Dim(1), f.Up, f.Down, int(cfg.NumActiveExperts), cfg.RoutedScalingFactor, T)
		r = mlx.OlalaGEMV([]*mlx.Array{parts, hcat}, []*mlx.OlalaWeight{f.OutCat}, T, f.InCat.Rows,
			mlx.OlalaGEMVOptions{Prologue: mlx.OlalaReLU2Cat, Ka: dLatent, Sa: dLatent, Sb: hcat.Dim(1), StatsX: resid})
		g, gst = r[0], r[1]
		prevGeo = f.GeoMlp
	}
	return mlx.Reshape(mlx.OlalaGeodesic(resid, g, prevGeo, T, D), 1, 1, int32(D))
}

package lfm3speech

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestFSQRoundTrip(t *testing.T) {
	cfg, err := ParseCodecConfig([]byte(codecConfig))
	if err != nil {
		t.Fatal(err)
	}
	c := &Codec{cfg: cfg}
	frame := make([]int32, cfg.NumCodebooks)
	for code := range int32(cfg.CodebookSize()) {
		frame[3] = code
		latents, err := c.fsq([][]int32{frame})
		if err != nil {
			t.Fatal(err)
		}
		got, radix := 0, 1
		for i, l := range cfg.FSQLevels {
			v := latents[3*len(cfg.FSQLevels)+i]
			if v < -1 || v > 1 {
				t.Fatalf("code %d latent %v outside [-1, 1]", code, v)
			}
			half := float64(l / 2)
			got += int(math.Round(float64(v)*half+half)) * radix
			radix *= l
		}
		if got != int(code) {
			t.Fatalf("code %d decodes to digits of %d", code, got)
		}
	}
	if _, err := c.fsq([][]int32{make([]int32, cfg.NumCodebooks-1)}); err == nil {
		t.Fatal("accepted a short frame")
	}
	frame[0] = int32(cfg.CodebookSize())
	if _, err := c.fsq([][]int32{frame}); err == nil {
		t.Fatal("accepted the EOS code")
	}
}

func hann(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

func TestSynthesisBasisIsWindowedIRFFT(t *testing.T) {
	const n = 20
	window := hann(n)
	basis := synthesisBasis(window)
	bins := n/2 + 1
	rng := rand.New(rand.NewPCG(3, 4))
	spec := make([]complex128, bins)
	for k := range spec {
		spec[k] = complex(rng.NormFloat64(), rng.NormFloat64())
	}
	for j := range n {
		// Hermitian inverse DFT, ignoring the DC and Nyquist imaginary parts.
		var want complex128
		for k := range n {
			var x complex128
			switch {
			case k == 0 || k == n/2:
				x = complex(real(spec[k]), 0)
			case k < n/2:
				x = spec[k]
			default:
				x = cmplx.Conj(spec[n-k])
			}
			want += x * cmplx.Exp(complex(0, 2*math.Pi*float64(k*j)/n))
		}
		w := real(want) / n * float64(window[j])
		var got float64
		for k := range bins {
			got += real(spec[k])*float64(basis[k*n+j]) + imag(spec[k])*float64(basis[(bins+k)*n+j])
		}
		if math.Abs(got-w) > 1e-5 {
			t.Fatalf("sample %d: basis gives %v, inverse DFT %v", j, got, w)
		}
	}
}

func TestOverlapAddStreams(t *testing.T) {
	cfg := &CodecConfig{WinLength: 16, HopLength: 4, FrameStack: 2}
	window := hann(16)
	rng := rand.New(rand.NewPCG(5, 6))
	const frames = 14
	all := make([]float32, frames*16)
	for i := range all {
		all[i] = float32(rng.NormFloat64())
	}
	whole := newOverlapAdd(cfg, window)
	want := append(whole.add(all), whole.finish()...)
	if len(want) != frames*4 {
		t.Fatalf("decoded %d samples, want %d", len(want), frames*4)
	}
	// Direct definition: sum the frames, divide by the summed squared window,
	// trim (win-hop)/2 from each end.
	for i := range want {
		n := i + 6
		var num, den float64
		for f := range frames {
			if j := n - 4*f; j >= 0 && j < 16 {
				num += float64(all[f*16+j])
				den += float64(window[j]) * float64(window[j])
			}
		}
		if math.Abs(float64(want[i])-num/den) > 1e-4*math.Max(1, math.Abs(num/den)) {
			t.Fatalf("sample %d = %v, want %v", i, want[i], num/den)
		}
	}
	for _, chunk := range []int{2, 4, 6} {
		o := newOverlapAdd(cfg, window)
		var got []float32
		for f := 0; f < frames; f += chunk {
			got = append(got, o.add(all[f*16:min(frames, f+chunk)*16])...)
		}
		got = append(got, o.finish()...)
		if len(got) != len(want) {
			t.Fatalf("chunk %d: %d samples, want %d", chunk, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("chunk %d sample %d = %v, want %v", chunk, i, got[i], want[i])
			}
		}
	}
}

// tinyCodec builds a random LiquidCodec decoder with the published layer
// pattern at toy size.
func tinyCodec(t *mlxtest.T) *Codec {
	rng := rand.New(rand.NewPCG(7, 8))
	ts := map[string]*mlx.Array{}
	put := func(name string, scale float64, shape ...int) {
		ts[name] = randArray(rng, scale, shape...)
	}
	ones := func(name string, n int) {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + 0.1*rng.NormFloat64())
		}
		ts[name] = mlx.FromValues(v, n)
	}
	put("quantizer.project_out.weight", 0.5, 4, 4)
	put("quantizer.project_out.bias", 0.1, 4)
	put("from_quantizer.weight", 0.5, 8, 4)
	put("from_quantizer.bias", 0.1, 8)
	ones("decoder.lfm.embedding_norm.weight", 8)
	for i := range 4 {
		p := fmt.Sprintf("decoder.lfm.layers.%d.", i)
		for _, n := range []string{"operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"} {
			ones(p+n+".weight", 8)
		}
		put(p+"feed_forward.w1.weight", 0.3, 12, 8)
		put(p+"feed_forward.w3.weight", 0.3, 12, 8)
		put(p+"feed_forward.w2.weight", 0.3, 8, 12)
		if i%2 == 0 {
			put(p+"conv.conv.weight", 0.5, 8, 1, 3)
			put(p+"conv.in_proj.weight", 0.3, 24, 8)
			put(p+"conv.out_proj.weight", 0.3, 8, 8)
		} else {
			put(p+"self_attn.q_proj.weight", 0.3, 8, 8)
			put(p+"self_attn.k_proj.weight", 0.3, 4, 8)
			put(p+"self_attn.v_proj.weight", 0.3, 4, 8)
			put(p+"self_attn.o_proj.weight", 0.3, 8, 8)
			put(p+"self_attn.gate_proj.weight", 0.3, 2, 8)
			ones(p+"self_attn.q_norm.weight", 4)
			ones(p+"self_attn.k_norm.weight", 4)
		}
	}
	put("decoder.lin.weight", 0.2, 18, 8)
	put("decoder.lin.bias", 0.1, 18)
	ts["decoder.istft.window"] = mlx.FromValues(hann(16), 16)
	const config = `{"model_type": "liquid_codec", "encoder_dim": 8, "stft_n_fft": 16, "stft_hop_length": 4,
	 "stft_win_length": 16, "stft_frame_stack": 2, "quantizer_dim": 4, "fsq_levels": [3, 2], "n_q": 2,
	 "decoder_window": 3, "sampling_rate": 24000}`
	c, err := LoadCodec(ts, "", []byte(config))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func randArray(rng *rand.Rand, scale float64, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return mlx.FromValues(v, shape...)
}

func TestCodecStreamMatchesDecode(t *testing.T) {
	// Compare in full float32; TF32 matmuls (the M5 default) differ by shape.
	t.Setenv("MLX_ENABLE_TF32", "0")
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := tinyCodec(t)
		rng := rand.New(rand.NewPCG(9, 10))
		codes := make([][]int32, 11)
		for i := range codes {
			codes[i] = []int32{rng.Int32N(6), rng.Int32N(6)}
		}
		want, err := c.Decode(codes)
		if err != nil {
			t.Fatal(err)
		}
		if len(want) != len(codes)*c.Config().FrameSamples() {
			t.Fatalf("decoded %d samples, want %d", len(want), len(codes)*c.Config().FrameSamples())
		}
		var peak float64
		for _, v := range want {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		for _, chunk := range []int{1, 2, 5} {
			s := c.NewStream()
			var got []float32
			for i := 0; i < len(codes); i += chunk {
				out, err := s.Push(codes[i:min(len(codes), i+chunk)]...)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, out...)
			}
			got = append(got, s.Flush()...)
			s.Close()
			if len(got) != len(want) {
				t.Fatalf("chunk %d: %d samples, want %d", chunk, len(got), len(want))
			}
			// Float32 accumulation order differs with chunk shape (114-145 dB
			// on M3 and M5); a streaming bug would be far below this.
			var signal, noise float64
			for i := range got {
				d := float64(got[i] - want[i])
				signal += float64(want[i]) * float64(want[i])
				noise += d * d
			}
			if snr := 10 * math.Log10(signal/max(noise, 1e-30)); snr < 100 {
				t.Fatalf("chunk %d: streaming SNR %.1f dB, want >= 100", chunk, snr)
			} else {
				t.Logf("chunk %d: streaming SNR %.1f dB", chunk, snr)
			}
		}
	})
}

package inkling2

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Short target forwards cover the anchor and up to seven draft proposals.
const shortTargetRows = 8

// projectTarget gives each row the same result in short prefill, ordinary
// decode, and draft verification. Different MLX kernels round BF16 results
// differently, and the routed experts can amplify that difference. Short
// products compute each row independently; without them, tiles are padded to
// one matrix shape. Flatten requests before partitioning the rows so joining a
// batch cannot change the kernel either. Larger prefill forwards keep the
// usual projection path.
func projectTarget(projection *nn.Linear, x *mlx.Array) *mlx.Array {
	if x.NumDims() != 3 || x.Dim(1) > shortTargetRows {
		return projection.Forward(x)
	}
	rows, tokens, hidden := x.Dim(0), x.Dim(1), x.Dim(2)
	total := rows * tokens
	if total == 0 {
		return projection.Forward(x)
	}
	x = x.Reshape(total, hidden)
	var results []*mlx.Array
	for begin := 0; begin < total; begin += shortTargetRows {
		end := min(total, begin+shortTargetRows)
		n := end - begin
		tile := x
		if n < total {
			tile = x.Slice(mlx.Slice(begin, end), mlx.Slice())
		}
		if projection.Bias == nil {
			if result := mlx.ShortMatmul(tile, projection.Weight); result != nil {
				results = append(results, result)
				continue
			}
		}
		if n < shortTargetRows {
			tile = mlx.PadConstant(tile, []int{0}, []int{0}, []int{shortTargetRows - n})
		}
		result := projection.Forward(tile)
		if n < shortTargetRows {
			result = result.Slice(mlx.Slice(0, n), mlx.Slice())
		}
		results = append(results, result)
	}
	out := results[0]
	if len(results) > 1 {
		out = mlx.Concatenate(results, 0)
	}
	return out.Reshape(rows, tokens, out.Dim(1))
}

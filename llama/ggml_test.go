package llama

import "testing"

func TestGgmlBasic(t *testing.T) {
	// This test replicates the comment in the ggml.h showing basic usage

	expected := float32(16.0)
	// For example, here we define the function: f(x) = a*x^2 + b
	//
	// struct ggml_init_params params = {
	//   .mem_size   = 16*1024*1024,
	//   .mem_buffer = NULL,
	// };

	params := NewGGMLInitParams(16 * 1024 * 1024)

	// // memory allocation happens here
	// struct ggml_context * ctx = ggml_init(params);
	ctx := GGMLInit(params)
	if err := ctx.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	x := GGMLNewTensor1d(ctx, GGML_TYPE_F32, 1)
	if err := x.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// ggml_set_param(ctx, x); // x is an input variable
	GGMLSetParam(ctx, x)

	// struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	a := GGMLNewTensor1d(ctx, GGML_TYPE_F32, 1)
	if err := a.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	b := GGMLNewTensor1d(ctx, GGML_TYPE_F32, 1)
	if err := b.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
	x2 := GGMLMul(ctx, x, x)
	if err := x2.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
	f := GGMLAdd(ctx, GGMLMul(ctx, a, x2), b)
	if err := f.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// Notice that the function definition above does not involve any actual computation. The computation is performed only
	// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
	//
	// struct ggml_cgraph * gf = ggml_new_graph(ctx);
	gf := GGMLNewGraph(ctx)
	if err := gf.GetError(); err != nil {
		t.Fatal(err.Error())
	}

	// ggml_build_forward_expand(gf, f);
	GGMLBuildForwardExpand(gf, f)

	// // set the input variable and parameter values
	// ggml_set_f32(x, 2.0f);

	GGMLSetF32(x, 2.0)

	// ggml_set_f32(a, 3.0f);
	GGMLSetF32(a, 3.0)

	// ggml_set_f32(b, 4.0f);
	GGMLSetF32(b, 4.0)

	// ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
	GGMLGraphComputeWithCtx(ctx, gf, 4)

	// printf("f = %f\n", ggml_get_f32_1d(f, 0));
	resp := GGMLGetF32_1d(f, 0)
	if resp != expected {
		t.Fatalf("expected %f got %f", expected, resp)
	}
}

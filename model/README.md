# Ollama Models

Large Language Models in Ollama are defined in the Go programming language within this directory.


## Model Implementation Guide

Ollama supports multiple backends, and provides an astracted interface for model implementers.  [Backend API ](../ml/backend.go)

This API is designed to be similar to other popular python libraries such as
PyTorch, with row-major tensors and a forward function that takes a sequence of
inputs.  

Use an existing model as an initial reference, such as [llama](./models/llama/)

Cheatsheet:

<table>
<tr>
  <td><b>PyTorch</b></td>
  <td><b>Ollama</b></td>
</tr>
<tr>
  <td>torch.zeros((2, 2))</td>
  <td>ctx.Zeros(ml.DTypeF32, 2, 2)</td>
</tr>
<tr>
  <td>tensor.view((2, 2))</td>
  <td>t.Reshape(ctx, 2, 2)</td>
</tr>
<tr>
   <td>torch.permute(t1, (1, 2, 3))</td>
   <td>t1.Permute(ctx, 1, 2, 3)</td>
</tr>
<tr>
    <td>torch.add(t1, t2)</td>
    <td>t1.Add(ctx, t2)</td>
</tr>
<tr>
<td>

```python
class Attention(nn.Module):
    def __call__(self, ...):
        ...
```

</td>
<td>

```go
func (sa *SelfAttention) Forward(ctx ml.Context,
                                 hiddenState, positionIDs ml.Tensor,
                                 cache kvcache.Cache,
                                 opts *Options) ml.Tensor {
    ...
}
```

</td>
</tr>
</table>
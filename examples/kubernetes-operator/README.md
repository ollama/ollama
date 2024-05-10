# Deploy Ollama to Kubernetes Operator


### Prerequisites

- Install [Ollama-operator](https://github.com/nekomeowww/ollama-operator)

### Steps

1. create a `Model` CR, serving model `phi` as an example:

   ```
   kubectl apply -f - << EOF
   apiVersion: ollama.ayaka.io/v1
   kind: Model
   metadata:
     name: phi
   spec:
     image: phi
   EOF
   ```

2. forward the ports to access the model outside cluster:
   ```
   kubectl port-forward svc/ollama-model-phi ollama
   ```

3. Install [ollama cli](https://github.com/ollama/ollama) and interact with the model with either API or ollama CLI( e.g. `ollama run phi` ).

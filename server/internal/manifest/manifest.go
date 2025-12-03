// Package manifest provides documentation for the Ollama manifest format.
// This package contains no code.
//
// # Manifests
//
// A manifest is a JSON object that describes a model. The JSON object has a
// single field "layers" which is a list of layers that make up the model.
// A layer is a single, logical unit of a model. Layers are stored in the cache
// as files with the name of the digest of the layer. Layers are pushed and
// pulled from the registry as blobs.
//
// A layer is represented as a JSON object with the following fields:
//
//   - "digest": The digest of the layer.
//   - "mediaType": The media type of the layer.
//   - "size": The size of the layer in bytes.
//
// Layers are typically stored in a blob store, such as a registry, and are
// referenced by their digest. This package does not define how layers are
// stored or retrieved.
//
// # Configuration Layer
//
// The configuration of a model is represented as a layer with the media type:
//
//	application/vnd.ollama.image.config; type=<type>
//
// The "type" parameter in the media type specifies the format of the
// configuration (e.g., "safetensor" or "gguf").
//
// There may be only one configuration layer in a model.
//
// # Template Layer
//
// The model template is a layer with the media type:
//
//	application/vnd.ollama.image.template; [name=<name>]
//
// The "name" parameter in the media type specifies the name of the template as
// for lookup at runtime. The name is optional and may be omitted. If omitted,
// the template is the default template for the model.
//
// # Tensor Layers
//
// The tensors of a model are represented as layers with the media type:
//
//	application/vnd.ollama.image.tensor; name=<name>; dtype=<dtype>; shape=<shape>
//
// The "name" parameter in the media type specifies the name of the tensor as
// defined in the model's configuration and are bound only by the rules for
// names as defined in the configuration format, as represented by the
// configuration's "type".
//
// The "dtype" parameter in the media type specifies the data type of the tensor
// as a string.
//
// TODO: Define more specifically how to represent data types as strings.
//
// The "shape" parameter in the media type specifies the shape of the tensor as
// a comma-separated list of integers; one per dimension.
//
// # Tokenization Layers
//
// The tokenization of a model is represented as a layer with the media type:
//
//	application/vnd.ollama.image.tokenizer
//
// The configuration of the tokenizer is represented as a layer with the media type:
//
//	application/vnd.ollama.image.tokenizer.config
//
// # Miscellaneous Layers
//
// These extra layer mime types are reserved:
//
//	application/vnd.ollama.image.license
//
// This layer contains one of the many licenses for the model in plain text.
//
// # Example Manifest
//
// The following is an example manifest containing a configuration, a model
// template, and two tensors (digests shortened for brevity):
//
//	{
//	  "layers": [{
//	      "digest": "sha256:a...",
//	      "mediaType": "application/vnd.ollama.image.config; type=safetensors",
//	      "size": 1234
//	    },{
//	      "digest": "sha256:b...",
//	      "mediaType": "application/vnd.ollama.image.template",
//	      "size": 5678
//	    },{
//	      "digest": "sha256:c...",
//	      "mediaType": "application/vnd.ollama.image.tensor; name=input; dtype=F32; shape=1,2,3",
//	      "size": 9012
//	    },{
//	      "digest": "sha256:d...",
//	      "mediaType": "application/vnd.ollama.image.tensor; name=output; dtype=I32; shape=4,5,6",
//	      "size": 3456
//	  }]
//	}
//
// # Legacy Media Types
//
// The appliaction/vnd.ollama.image.model media type is deprecated, but will
// remain supported for backwards compatibility, for some undefined amount of
// time. New models should use the media types defined above.
//
// # Reserved media types
//
// The media type prefix "application/vnd.ollama.image." is reserved for
// defining new media types for layers known to Ollama. Currently, all other
// prefixes are ignored by official Ollama registry clients.
package manifest

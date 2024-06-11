# onnxruntime_transformers
transformers for production runtime, 3x faster on cpu, no pytorch nor tensorflow included

## convert models to onnx
install converter
`pip install optimum[exporters]`

convert embedding model to onnx
`optimum-cli export onnx --task sentence-similarity --model "infgrad/stella-base-zh-v3-1792d" bert_embed`

convert sentence correction model to onnx
`optimum-cli export onnx --task fill-mask --model "shibing624/macbert4csc-base-chinese" bert_csc`

convert ner model to onnx
`optimum-cli export onnx --task token-classification --model "shibing624/bert4ner-base-chinese" bert_ner`

## inference with onnx
generate embeddings
```
from onnxruntime_transformers import OnnxruntimeTransformers
encoder = OnnxruntimeTransformers("./bert_embed/tokenizer.json", "./bert_embed/model.onnx")
embeddings = encoder.encode([
    "how are you",
    "I'm fine thank you, and you?",
])
```
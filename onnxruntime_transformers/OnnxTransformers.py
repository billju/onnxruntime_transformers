from typing import List
import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer


class OnnxruntimeTransformers:
    def __init__(self, tokenizer_json: str, model_onnx: str):
        self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_json)
        self.session = InferenceSession(model_onnx)

    def tokenize(self, sentences: List[str]) -> dict:
        encoded = self.tokenizer.encode_batch(sentences)
        max_len = max(len(e) for e in encoded)
        for e in encoded:
            e.pad(max_len)
        inputs = dict(
            input_ids=np.array([e.ids for e in encoded], dtype=np.int64),
            attention_mask=np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            ),
        )
        if "token_type_ids" in [i.name for i in self.session.get_inputs()]:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        return inputs

    def encode(self, sentences: List[str]) -> np.array:
        inputs = self.tokenize(sentences)
        token_embeddings = self.session.run(["token_embeddings"], inputs)[0]
        mask_expanded = np.broadcast_to(
            np.expand_dims(inputs["attention_mask"], axis=-1), token_embeddings.shape
        )
        return np.sum(token_embeddings * mask_expanded, axis=1) / np.clip(
            np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None
        )

    def forward(self, sentences: List[str]) -> np.array:
        logit = self.session.run(["logits"], self.tokenize(sentences))[0]
        return logit

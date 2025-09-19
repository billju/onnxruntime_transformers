from typing import List
import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

def recursive_split(text: str, separators=['。', '，'], chunk_size=256):
    if len(text) <= chunk_size:
        return [text]
    indices = [i for i, char in enumerate(text) if char in separators]
    if len(indices) == 0:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    mid = indices[len(indices) // 2]
    return recursive_split(text[: mid]) + recursive_split(text[mid + 1 :])

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

    def csc(self, text: str, batch_size=8, threshold=0.7) -> str:
        chunks = recursive_split(text)
        logits = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logits.extend(self.forward(batch))
        idx_chr, i = {}, 0
        for chunk, logit in zip(chunks, logits):
            # softmax轉換機率
            e_x = np.exp(logit - np.max(logit, axis=-1, keepdims=True))
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            # 去掉頭[CLS]尾[SEP]符號
            probs = probs[1:-1]
            max_probs = np.max(probs, axis=-1)
            max_ids = np.argmax(probs, axis=-1)
            # 文字轉token比對
            chunk_tokens = self.tokenizer.encode(chunk).tokens[1:-1]
            # 替換[UNK]成原始文字
            j, k, s = 0, 0, ''
            while j < len(chunk) and k < len(chunk_tokens):
                token = chunk_tokens[k].replace('#', '')
                if chunk[j] in ['\n', ' ']:
                    j += 1
                elif token == '[UNK]':
                    # 收集 UNK 對應的原始文字直到找到下一個匹配的 token
                    current_unk = ''
                    while j < len(chunk):
                        char=chunk_tokens[k + 1].replace('#', '')
                        if k + 1 < len(chunk_tokens) and chunk[j:].startswith(char):
                            break
                        current_unk += chunk[j]
                        j += 1
                    chunk_tokens[k] = current_unk  # 將 UNK 替換為原始文字
                    k += 1
                elif chunk.startswith(token, j):
                    if s:
                        chunk_tokens[k - 1] = s
                        s = ''
                    j += len(token)
                    k += 1
                else:
                    s += chunk[j]
                    j += 1
            # 將id轉換成token
            correct_tokens = [self.tokenizer.id_to_token(id) for id in max_ids]
            for o, c, p in zip(chunk_tokens, correct_tokens, max_probs):
                # o: 原始token (original)
                # c: 修正後的token (corrected)
                # p: 該修正的概率值 (probability)
                if o != c and p >= threshold and not o.isascii() and not c.isascii():
                    idx_chr[i] = c
                i += len(o.replace('#', ''))
        # 跳過換行、空白符號
        corrected, i = '', 0
        for c in text:
            corrected += idx_chr[i] if i in idx_chr else c
            i += 0 if c in ['\n', ' '] else 1
        return corrected

    def ner(self, text: str, labels=['I-ORG','B-LOC','O','B-ORG','I-LOC','I-PER','B-TIME','I-TIME','B-PER']):
        entity= []
        chunks = recursive_split(text)
        idss = []
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            idss.extend(self.forward(chunks[i : i + batch_size]).argmax(2))
        for chunk, ids in zip(chunks, idss):
            tokens = self.tokenizer.encode(chunk).tokens
            name, label = '', ''
            for id, token in zip(ids, tokens):
                if labels[id].startswith('B'):
                    if labels[id].split('-').pop() == label:
                        name += token.strip('#')
                    else:
                        if name:
                            entity.append([name, label])
                        name=token.strip('#')
                        label=labels[id].split('-').pop()
                elif labels[id].startswith('I'):
                    name += token.strip('#')
                elif name:
                    entity.append([name, label])
                    name, label = '', ''
        return entity
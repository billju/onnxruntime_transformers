from .OnnxruntimeTransformers import OnnxruntimeTransformers

def recursive_split(text: str, separators=["。", "，"], chunk_size=256):
    if len(text) <= chunk_size:
        return [text]
    indices = [i for i, char in enumerate(text) if char in separators]
    if len(indices) == 0:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    mid = indices[len(indices) // 2]
    return recursive_split(text[: mid]) + recursive_split(text[mid + 1 :])
"""Helper functions to generate embeddings for a given text."""

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from util.config import config

_torch_backend = "cpu"
if torch.cuda.is_available():
    _torch_backend = "cuda"
if torch.backends.mps.is_available():
    _torch_backend = "mps"

device = torch.device(_torch_backend)

_tokenizers: dict[str, Any] = {}
_models: dict[str, Any] = {}


def _lazy_get_model_and_tokenizer(model_name: str):
    if model_name not in _tokenizers or model_name not in _models:
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, cache_dir=config.MODEL_CACHE_DIR)
        _models[model_name] = AutoModel.from_pretrained(model_name, cache_dir=config.MODEL_CACHE_DIR).to(device)
    return _tokenizers[model_name], _models[model_name]


models_for_384_embeddings = ["sentence-transformers/all-MiniLM-L12-v2"]
models_for_768_embeddings = [
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "olm-roberta-base-dec-2022",
]


def generate_embedding(text: str, model: str) -> list[float]:
    """Generate an embedding for a given text using a pre-trained model.

    :param text: The text to embed. Must be less than 512 tokens long.
    :param model:
        The name of the model to use.

        for embeddings of length 384 use:
        - sentence-transformers/all-MiniLM-L12-v2

        for embeddings of length 768 use:
        - sentence-transformers/multi-qa-mpnet-base-dot-v1
        - olm-roberta-base-dec-2022
    :return: The embedding vector. A list of floats of length v_len.
    """
    tokenizer, model = _lazy_get_model_and_tokenizer(model)
    tokenized = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        output = model(**tokenized).pooler_output.cpu().numpy().tolist()
        return output[0]

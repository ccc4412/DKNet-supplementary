# -*- coding: utf-8 -*-
"""句子编码器：官方 GUSE/bert-base-uncased 分支的中文等价替换。

- BertZhEncoder: bert-base-chinese（torch），取 pooler_output（与官方
  TFBertModel(output_hidden_states=True)[1] 同一语义），768 维。
- MockEncoder: 确定性字符 bigram 哈希向量，仅用于无权重/无 GPU 环境下的
  流水线冒烟测试，不用于论文实验。
"""
import hashlib

import numpy as np


class MockEncoder:
    dim = 768
    name = "mock"

    @staticmethod
    def _vec(text):
        v = np.zeros(768, dtype=np.float32)
        s = text.replace(" ", "")
        grams = [s[i:i + 2] for i in range(max(1, len(s) - 1))] or [s]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % 768
            sign = 1.0 if (h >> 16) & 1 else -1.0
            v[idx] += sign
            v[(h >> 8) % 768] += sign * 0.5
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, sentences, batch_size=64):
        return np.stack([self._vec(s) for s in sentences])


class BertZhEncoder:
    dim = 768
    name = "bert-base-chinese"

    def __init__(self, device=None):
        import torch
        from transformers import BertModel, BertTokenizerFast
        self.torch = torch
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-chinese")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def encode(self, sentences, batch_size=64):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                enc = self.tokenizer(sentences[i:i + batch_size], truncation=True,
                                     padding=True, max_length=100, return_tensors="pt").to(self.device)
                pooled = self.model(**enc).pooler_output
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def get_encoder(name=None):
    from common import zh_config as cfg
    name = name or cfg.ENCODER
    return BertZhEncoder() if name == "bert" else MockEncoder()

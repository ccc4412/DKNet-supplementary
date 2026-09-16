# -*- coding: utf-8 -*-
"""官方 model_BERT.py 的 PyTorch 等价实现。

架构与训练日程逐项对应官方 Keras 版本（偏离 D6: TF/Keras -> PyTorch, 结构不变）:
  bert-base-uncased -> bert-base-chinese（偏离 D4）
  BERT pooler(768) -> Dense(768,relu) -> Dense(512,relu) -> Dense(341,relu) -> Dense(150)
  loss = cosine_distance (1 - cos); 两阶段: 迁移(冻结编码器, Adam 1e-3) -> 全量微调(3e-5)
  EarlyStopping(val_loss, patience=3), MAX_EPOCHS=50, BATCH=32, checkpoint 存最优
"""
import os

import numpy as np
import torch
import torch.nn as nn

from common import zh_w2v as emb
from common.zh_config import BERT_MODEL_NAME, ENCODER_DIM
from hashet_zh import constants as c

SCALE_1 = 2 / 3  # 官方隐藏层缩放系数


class MockBackbone:
    """冒烟测试用确定性编码骨干（仅用于打通流水线, 不用于论文实验）。"""
    name = "mock"
    dim = ENCODER_DIM

    @staticmethod
    def _vec(text):
        import hashlib
        v = np.zeros(ENCODER_DIM, dtype=np.float32)
        s = text.replace(" ", "")
        grams = [s[i:i + 2] for i in range(max(1, len(s) - 1))] or [s]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            v[h % ENCODER_DIM] += 1.0 if (h >> 16) & 1 else -1.0
            v[(h >> 8) % ENCODER_DIM] += 0.5 if (h >> 24) & 1 else -0.5
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def encode(sentences):
        return np.stack([MockBackbone._vec(s) for s in sentences])


class HashetNet(nn.Module):
    """bert-base-chinese + 官方 MLP 头; use_mock 时以 MockBackbone 特征替代 BERT。"""

    def __init__(self, out_dim, use_mock=False):
        super().__init__()
        self.use_mock = use_mock
        if not use_mock:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
        h = ENCODER_DIM
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, int(h * SCALE_1)), nn.ReLU(),
            nn.Linear(int(h * SCALE_1), int(h * SCALE_1 * SCALE_1)), nn.ReLU(),
            nn.Linear(int(h * SCALE_1 * SCALE_1), out_dim),
        )

    def forward(self, feats):
        return self.head(feats)

    def encode(self, sentences, device, batch_size=64):
        """文本 -> pooler/缓存特征 (N,768), 已 detach 之前是否需要梯度由调用方决定。"""
        if self.use_mock:
            return torch.as_tensor(MockBackbone.encode(sentences), device=device)
        toks = self._tok(sentences, truncation=True, padding=True, max_length=100,
                         return_tensors="pt").to(device)
        return self.bert(**toks).pooler_output

    def init_tokenizer(self):
        if not self.use_mock:
            from transformers import BertTokenizerFast
            self._tok = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME, local_files_only=True)


def _cosine_loss(pred, target):
    return (1 - nn.functional.cosine_similarity(pred, target, dim=-1)).mean()


class HashetModel:
    """两阶段训练 + top-k 预测, 对外接口对齐官方 model_BERT.transfer_and_fine_tune /
    predict_top_k_hashtags。"""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock = c.ENCODER_BACKEND == "mock"
        self.net = HashetNet(c.LATENT_SPACE_DIM, use_mock=self.use_mock).to(self.device)
        self.net.init_tokenizer()
        self.trained = False

    def _feats(self, sentences, need_grad):
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx:
            if self.use_mock:
                return self.net.encode(sentences, self.device)
            self.net.bert.train(need_grad)
            return self.net.encode(sentences, self.device)

    def _epoch_pass(self, feats, targets, optimizer, need_grad):
        perm = torch.randperm(len(feats), device=self.device)
        total, nb = 0.0, 0
        for i in range(0, len(perm), c.BATCH_SIZE):
            idx = perm[i:i + c.BATCH_SIZE]
            pred = self.net(feats[idx])
            loss = _cosine_loss(pred, targets[idx])
            if need_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            nb += 1
        return total / max(nb, 1)

    def _epoch_pass_bert(self, sentences, optimizer):
        """FT 阶段: 逐 batch 编码(BERT 参与训练) -> 前向 -> 反向。
        不能整 epoch 一次性前向: 共享编码图会在第二个 batch 反向时被二次穿越。"""
        perm = torch.randperm(len(sentences))
        total, nb = 0.0, 0
        for i in range(0, len(perm), c.BATCH_SIZE):
            batch = [sentences[j] for j in perm[i:i + c.BATCH_SIZE]]
            feats = self._feats(batch, need_grad=True)
            pred = self.net(feats)
            loss = _cosine_loss(pred, self.y_train[perm[i:i + c.BATCH_SIZE]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            nb += 1
        return total / max(nb, 1)

    def _save_weights(self, state, weights_path):
        """Windows 下直接 torch.save 大文件易被句柄/杀软锁住: 先写临时文件再替换。"""
        tmp = weights_path + ".tmp"
        for p in (weights_path, tmp):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        torch.save(state, tmp)
        os.replace(tmp, weights_path)

    def _run_stage(self, name, epochs, lr, weights_path):
        """统一两阶段实现: freeze = (name=='TL') 时冻结 BERT 参数。"""
        freeze = name == "TL"
        if not self.use_mock:
            self.net.bert.requires_grad_(not freeze)
        params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        best, bad = float("inf"), 0
        # TL 阶段编码器冻结 -> 特征只算一次; FT 阶段编码器参与训练 -> 每个 epoch 重算
        feats_train = self._feats(self.sentences_train, need_grad=False).detach() if freeze \
            else None
        feats_val = self._feats(self.sentences_val, need_grad=False).detach()
        for epoch in range(epochs):
            if freeze:
                tl = self._epoch_pass(feats_train, self.y_train, optimizer,
                                      need_grad=False)
            else:
                tl = self._epoch_pass_bert(self.sentences_train, optimizer)
            with torch.no_grad():
                val = _cosine_loss(self.net(feats_val), self.y_val).item()
            marker = ""
            if val < best:
                best, bad = val, 0
                self._save_weights(self.net.state_dict(), weights_path)
                marker = " *"
            else:
                bad += 1
            print(f"  [{name}] epoch {epoch+1}/{epochs} loss={tl:.4f} val={val:.4f}{marker}",
                  flush=True)
            if bad >= c.PATIENCE:
                print(f"  [{name}] early stopping (patience={c.PATIENCE})")
                break

    def transfer_and_fine_tune(self, sentences_train, sentences_test,
                               targets_train, targets_test):
        print("BERT version (PyTorch)", "| backbone:",
              "mock" if self.use_mock else BERT_MODEL_NAME)
        print("TRANSFER LEARNING STEP:")
        self.sentences_train = list(sentences_train)
        self.sentences_val = list(sentences_test)
        self.y_train = torch.as_tensor(np.asarray(targets_train), dtype=torch.float32,
                                       device=self.device)
        self.y_val = torch.as_tensor(np.asarray(targets_test), dtype=torch.float32,
                                     device=self.device)
        self._run_stage("TL", c.MAX_EPOCHS, c.LR_TRANSFER, c.TL_WEIGHTS)
        print("FINE TUNING STEP:")
        # 官方语义: 微调前载入迁移阶段最优权重
        self.net.load_state_dict(torch.load(c.TL_WEIGHTS, map_location=self.device))
        self._run_stage("FT", c.MAX_EPOCHS, c.LR_FINETUNE, c.FT_WEIGHTS)
        self.net.load_state_dict(torch.load(c.FT_WEIGHTS, map_location=self.device))
        self.trained = True

    def fine_tune_short(self, sentences_train, sentences_val, targets_train, targets_test,
                        ft_sentences=None, ft_targets=None, epochs=None):
        """H-ADAPTS rt_training 语义: 从当前权重继续, 在新窗口上渐进微调
        （官方 MAX_EPOCHS_FT=12, RT_LR=3e-5）。官方在 logic-window 主集与
        FT 窗口子集上联合微调; FT 子集包含于主集内, 这里直接以主集训练。"""
        self.sentences_train = list(sentences_train)
        self.sentences_val = list(sentences_val)
        self.y_train = torch.as_tensor(np.asarray(targets_train), dtype=torch.float32,
                                       device=self.device)
        self.y_val = torch.as_tensor(np.asarray(targets_test), dtype=torch.float32,
                                     device=self.device)
        self._run_stage("FT", epochs or c.MAX_EPOCHS_FT, c.RT_LR, c.FT_WEIGHTS)
        self.net.load_state_dict(torch.load(c.FT_WEIGHTS, map_location=self.device))
        self.trained = True

    def predict_top_k_hashtags(self, sentences, k, w2v=None):
        """官方语义: 预测向量 -> W2V 近邻 top-1000 -> retain_hts -> 前 k 个。"""
        assert self.trained
        self.net.eval()
        with torch.no_grad():
            feats = self._feats(list(sentences), need_grad=False)
            preds = self.net(feats).cpu().numpy()
        w2v = w2v or emb.load_word2vec(c.W2V_SAVE_FILE_NAME)
        return [emb.retain_hts(w2v.wv.similar_by_vector(v, topn=1000))[:k] for v in preds]

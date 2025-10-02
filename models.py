# models.py

from typing import List
from collections import Counter, defaultdict
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from sentiment_data import *  # noqa: F401,F403
import sentiment_data as _sd

# ---------------------------------------------------------------------
# Limit CPU threads so it does not use up all cores (reduces heat/noise).
#For local run to work properly

try:
    _threads = int(os.environ.get("DAN_THREADS", "1"))
    torch.set_num_threads(max(1, _threads))
    torch.set_num_interop_threads(max(1, _threads))
except Exception:
    pass


# ---------------------------------------------------------------------
# Base API used by the driver
# ---------------------------------------------------------------------
class SentimentClassifier(object):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        raise NotImplementedError

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ws, has_typos) for ws in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    """Baseline: always predict positive."""
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        return 1


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _words_to_ids(words: List[str], indexer) -> List[int]:
    """Map tokens to indices via Indexer; unknown -> UNK (1)."""
    ids = []
    for w in words:
        try:
            idx = indexer.index_of(w)
        except Exception:
            idx = -1
        ids.append(1 if idx == -1 else idx)
    return ids


def _pad_batch(batch_seqs: List[List[int]], pad_idx: int = 0) -> torch.LongTensor:
    """Right-pad a batch of index sequences to [B, T_max]."""
    if not batch_seqs:
        return torch.LongTensor([])
    max_len = max(len(x) for x in batch_seqs)
    B = len(batch_seqs)
    out = torch.full((B, max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch_seqs):
        if seq:
            out[i, :len(seq)] = torch.LongTensor(seq)
    return out


def _iter_vocab(indexer):
    """Yield vocabulary strings from the Indexer."""
    try:
        n = len(indexer)
        for i in range(n):
            try:
                yield indexer.get_object(i)
            except Exception:
                yield indexer.objs[i]
    except Exception:
        for w in getattr(indexer, "objs", []):
            yield w


def _build_prefix_map(indexer, k: int = 3):
    """prefix(first k chars) -> list of vocab words with that prefix."""
    pref = defaultdict(list)
    for w in _iter_vocab(indexer):
        if not isinstance(w, str):
            continue
        if w in ("PAD", "UNK"):
            continue
        if len(w) < k:
            continue
        pref[w[:k]].append(w)
    return pref


def _build_word_freq(train_exs: List["SentimentExample"], indexer) -> Counter:
    """Word frequency from training set (helps tie-break corrections)."""
    c = Counter()
    for ex in train_exs:
        for w in ex.words:
            try:
                if indexer.index_of(w) != -1:
                    c[w] += 1
            except Exception:
                pass
    return c


def _levenshtein_cutoff(a: str, b: str, maxd: int = 2) -> int:
    """Levenshtein distance with early cutoff (>maxd means 'too far')."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd:
        return maxd + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        start = max(1, i - maxd)
        end = min(lb, i + maxd)
        cur = [i] + [maxd + 1] * lb
        for j in range(start, end + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        if min(cur) > maxd:
            return maxd + 1
        prev = cur
    return prev[lb]


# Fallback: build embedding layer if the course helper isn't available
def _build_embedding_layer_fallback(word_embeddings, frozen: bool = True, padding_idx: int = 0):
    """
    Create nn.Embedding from the WordEmbeddings object.
    Tries to call the official helper if it exists, else constructs from known fields.
    """
    if hasattr(_sd, "get_initialized_embedding_layer"):
        return _sd.get_initialized_embedding_layer(word_embeddings, frozen=frozen)

    matrix = None
    # Common attribute names
    for attr in ("word_vectors", "vectors", "embeddings", "W", "weights", "word_vecs", "word_vecs_mat"):
        if hasattr(word_embeddings, attr):
            matrix = getattr(word_embeddings, attr)
            break

    # build by querying the indexer
    if matrix is None:
        indexer = getattr(word_embeddings, "word_indexer", getattr(word_embeddings, "indexer", None))
        if indexer is None:
            raise RuntimeError("Cannot locate indexer or vectors in word_embeddings.")
        dim = None
        for i in range(len(indexer)):
            w = indexer.get_object(i) if hasattr(indexer, "get_object") else indexer.objs[i]
            if w in ("PAD", "UNK"):
                continue
            if hasattr(word_embeddings, "get_embedding"):
                vec = word_embeddings.get_embedding(w)
                if vec is not None:
                    vec = np.asarray(vec, dtype=np.float32)
                    dim = vec.shape[-1]
                    break
        if dim is None:
            raise RuntimeError("Could not infer embedding dimension.")
        matrix = np.zeros((len(indexer), dim), dtype=np.float32)
        for i in range(len(indexer)):
            w = indexer.get_object(i) if hasattr(indexer, "get_object") else indexer.objs[i]
            if hasattr(word_embeddings, "get_embedding"):
                v = word_embeddings.get_embedding(w)
                if v is not None:
                    matrix[i] = np.asarray(v, dtype=np.float32)

    matrix = np.asarray(matrix, dtype=np.float32)
    emb = nn.Embedding(matrix.shape[0], matrix.shape[1], padding_idx=padding_idx)
    emb.weight.data.copy_(torch.from_numpy(matrix))
    emb.weight.requires_grad_(not frozen)
    return emb


# ---------------------------------------------------------------------
# Deep Averaging Network
# ---------------------------------------------------------------------
class DANNet(nn.Module):
    """Embed -> mean-pool (mask PAD) -> MLP -> logits(2)."""
    def __init__(self, emb_layer: nn.Embedding, emb_dim: int, hidden_size: int, dropout_p: float = 0.3):
        super().__init__()
        self.emb = emb_layer
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, batch_ids: torch.LongTensor) -> torch.Tensor:
        # batch_ids: [B, T]
        emb = self.emb(batch_ids)                         # [B, T, D]
        mask = (batch_ids != 0).float()                   # [B, T]
        lengths = mask.sum(dim=1).clamp_min(1.0)          # [B]
        avg = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)  # [B, D]
        h = self.dropout(self.act(self.fc1(avg)))
        return self.fc2(h)                                # [B, 2]


# ---------------------------------------------------------------------
# Classifier wrapper 
# ---------------------------------------------------------------------
class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, net: DANNet, indexer, device: str = "cpu",
                 typo_mode: bool = False, word_freq: Counter = None):
        self.net = net.to(device)
        self.indexer = indexer
        self.device = device
        self.typo_mode = bool(typo_mode)
        self.word_freq = word_freq or Counter()
        self.prefix_map = _build_prefix_map(indexer, k=3) if self.typo_mode else {}
        self._typo_cache = {}
        self.net.eval()

    def _correct_token(self, tok: str, k: int = 3, max_dist: int = 2, topk: int = 100) -> str:
        cand_list = self.prefix_map.get(tok[:k], [])
        if not cand_list:
            return tok
        tlen = len(tok)
        cand_list = sorted(cand_list, key=lambda w: self.word_freq.get(w, 0), reverse=True)[:topk]
        best_w, best_d, best_freq = tok, max_dist + 1, -1
        for w in cand_list:
            if abs(len(w) - tlen) > max_dist:
                continue
            d = _levenshtein_cutoff(tok, w, maxd=max_dist)
            if d <= max_dist:
                freq = self.word_freq.get(w, 0)
                if (d < best_d) or (d == best_d and freq > best_freq):
                    best_w, best_d, best_freq = w, d, freq
                    if d == 0:
                        break
        return best_w if best_d <= max_dist else tok

    def _maybe_fix_token(self, tok: str, k: int = 3) -> str:
        if not self.typo_mode or len(tok) < k:
            return tok
        try:
            if self.indexer.index_of(tok) != -1:
                return tok
        except Exception:
            pass
        hit = self._typo_cache.get(tok)
        if hit is not None:
            return hit
        corr = self._correct_token(tok, k=k, max_dist=2, topk=100)
        self._typo_cache[tok] = corr
        return corr

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        if has_typos and self.typo_mode:
            ex_words = [self._maybe_fix_token(w) for w in ex_words]
        ids = _words_to_ids(ex_words, self.indexer)
        batch = _pad_batch([ids])
        with torch.no_grad():
            logits = self.net(batch)
            return int(torch.argmax(logits, dim=1).item())

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        if has_typos and self.typo_mode:
            all_ex_words = [[self._maybe_fix_token(w) for w in ws] for ws in all_ex_words]
        batches = [_words_to_ids(ws, self.indexer) for ws in all_ex_words]
        batch_ids = _pad_batch(batches)
        with torch.no_grad():
            logits = self.net(batch_ids)
            return list(torch.argmax(logits, dim=1).cpu().numpy().tolist())


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _batch_iter(data, batch_size: int, shuffle: bool = True):
    idxs = list(range(len(data)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        yield [data[i] for i in idxs[start:start + batch_size]]


def train_model(args,
                train_exs: List["SentimentExample"],
                dev_exs: List["SentimentExample"],
                word_embeddings: "WordEmbeddings",
                indexer,
                train_model_for_typo_setting: bool = False) -> SentimentClassifier:
    """
    Train a DAN sentiment classifier (CPU). Designed to be fast by default on
    the driver's default CLI values, without modifying the driver.
    """
    device = "cpu"

    # Respect CLI if user changed it otherwise swap
    raw_hidden = getattr(args, "hidden_size", 100)
    raw_batch  = getattr(args, "batch_size", 1)
    raw_epochs = getattr(args, "num_epochs", 10)

    hidden_size = 64 if raw_hidden == 100 else int(raw_hidden)
    batch_size  = 64 if raw_batch == 1 else int(raw_batch)
    num_epochs  = 5  if raw_epochs == 10 else int(raw_epochs)

    lr = float(getattr(args, "lr", 1e-3))
    dropout_p = float(getattr(args, "dropout", 0.3))

    # Embedding layer
    emb_layer = _build_embedding_layer_fallback(word_embeddings, frozen=True, padding_idx=0)
    emb_dim = emb_layer.weight.shape[1]

    # Network / Optimizer / Loss
    net = DANNet(emb_layer, emb_dim=emb_dim, hidden_size=hidden_size, dropout_p=dropout_p)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    net.train()
    for epoch in range(num_epochs):
        total_loss, total_inst = 0.0, 0
        for batch in _batch_iter(train_exs, batch_size, shuffle=True):
            batch_ids = [_words_to_ids(ex.words, indexer) for ex in batch]
            x = _pad_batch(batch_ids)
            y = torch.LongTensor([ex.label for ex in batch])

            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            bs = len(batch)
            total_loss += float(loss.item()) * bs
            total_inst += bs

        avg_loss = total_loss / max(1, total_inst)
        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f}")

        # Inexpensive Lightweight dev eval
        if dev_exs:
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for b in _batch_iter(dev_exs, batch_size=256, shuffle=False):
                    ids = [_words_to_ids(ex.words, indexer) for ex in b]
                    x = _pad_batch(ids)
                    y = torch.LongTensor([ex.label for ex in b])
                    pred = torch.argmax(net(x), dim=1)
                    correct += int((pred == y).sum().item())
                    total += len(b)
                acc = 100.0 * correct / max(1, total)
                print(f"  dev_acc: {acc:.2f}%  ({correct}/{total})")
            net.train()

    # Wrap for inference
    word_freq = _build_word_freq(train_exs, indexer)
    clf = NeuralSentimentClassifier(net=net,
                                    indexer=indexer,
                                    device=device,
                                    typo_mode=train_model_for_typo_setting,
                                    word_freq=word_freq)
    return clf


# ---------------------------------------------------------------------
# Entry point expected by the driver (exact signature)
# ---------------------------------------------------------------------
def train_deep_averaging_network(args,
                                 train_exs: List["SentimentExample"],
                                 dev_exs: List["SentimentExample"],
                                 word_embeddings: "WordEmbeddings",
                                 train_model_for_typo_setting: bool) -> SentimentClassifier:
    """
    Wrapper with the exact name/signature the driver calls.
    Derives the indexer from word_embeddings and delegates to train_model.
    """
    # Best-effort: most course WordEmbeddings expose .word_indexer
    indexer = getattr(word_embeddings, "word_indexer", None)
    if indexer is None:
        # Fallbacks in case of different attribute names
        indexer = getattr(word_embeddings, "indexer", None)
    if indexer is None:
        raise RuntimeError("Could not locate the vocabulary indexer in word_embeddings.")

    return train_model(args, train_exs, dev_exs, word_embeddings, indexer, train_model_for_typo_setting)

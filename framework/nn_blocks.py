#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nn_blocks —— 序列模型族的**公共积木**（iTransformer / TimeXer / 前缀条件化三臂共用）。

它是什么
========
三种"神经"臂共享同一套数据契约与同一套编码器骨架，区别只在**分支组合**：

    ┌── 外生编码（plan-time 70 特征 → 24 等里程箱 → 每变量一个 token）
    │      VariableEncoder：共享 Linear → **变量维自注意力** → 融合 → 每变量一个向量
    ├── 内生（个性化）编码（前缀窗口 [s0, cut] 的 1 Hz 实测轨迹 → ≤64 patch）
    │      PatchEncoder：patch 投影 → **patch 维自注意力** → 池化 → 一个向量
    └── 融合（两种，可开关）
           cross（TimeXer 式）：外生序列当 context、内生当 query 做**前缀交叉注意力**
           concat（前缀条件化）：[外生表征 ⊕ 内生表征] 拼进残差头

**冻结基线作为骨架**（与源树 ``JJ_pathway`` / ``AA_prefixxer`` 同构，见各臂 README）::

    pred = baseline · exp(clamp(Δ, ±0.7))

其中 ``baseline`` 是**冻结的 SePT-70 plan-time 段级预测按趟求和**（残差/后缀两种口径），
只有 ``Δ``（与可选的 ``γ`` 门）被训练。这样做有两个硬好处：

1. **初始化即基线**：Δ=0 ⇒ 预测=基线。任何"训练把模型变差"的情形都能一眼看出来；
2. **口径可比**：Δ 是"前缀信息值多少个百分点"的**直接测量**，不是两次独立实验的差。

三处数值纪律（逐位复现的成败就在这里）
======================================
* **mask 即契约**：序列按 ``(n_trips, Lmax, F)`` 右填充；一切归约必须用 mask 加权，
  **不许**按批内最大长度重排（同一趟在不同 batch 会得到不同张量）。
* **只用前缀**：个性化分支的 patch 只覆盖 ``seg < prefix_len`` 的段（切断点之前）。
  ``prefix_len`` 由 ``features.build_cuts`` 的 ``n_pre`` 给出，**不手写算术**。
* **种子一种约定**：``apply_seed`` 同时驱动 torch / numpy / random；同种子重跑逐位一致。

本文件**不含 assert**：一切自检走显式 ``raise``。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

__all__ = [
    "NNError", "ExoConfig", "ExoEncoder", "PatchEncoder", "CrossFuse", "ConcatFuse",
    "PrefixNet", "bin_sequence", "prefix_lens_of", "patchify", "masked_mean",
    "PerVarScaler", "count_params", "train_seg_residual",
]

# 24 等里程箱（与源树的 ``S_fix=24`` 同构：把一趟切成 24 段等里程，每变量一个 token）
N_BINS = 24
PATCH_MAX = 64
PATCH_MIN_S = 30.0
MULT_CLAMP = 0.7          # pred = base · exp(clamp(Δ, ±0.7))


class NNError(RuntimeError):
    """神经臂的契约违反（形状 / mask / 前缀越界 / 分支死）。"""


# --------------------------------------------------------------------------- 张量工具
def masked_mean(x: Any, mask: Any, *, dim: int = 1) -> Any:
    """mask 加权均值（``mask`` 与 ``x`` 在 ``dim`` 上同长；权重 0/1）。"""
    import torch
    m = mask.to(dtype=x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    s = (x * m).sum(dim=dim)
    n = m.sum(dim=dim).clamp_min(1.0)
    return s / n


def count_params(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def cap_threads(n: int = 8) -> int:
    """把 torch 的 CPU 线程数**钉住**（默认 8；``DSH_TORCH_THREADS`` 可覆盖）。

    ★ 为什么必须钉：这些臂的张量很小（``d=24``、批 64），算力从来不是瓶颈，
      **线程调度**才是。实测：不限线程时 torch 会开满机器核数，与同机其它任务叠加后
      单步从 0.024 s 恶化到 30 s 量级（CPU 占用 1300%+，全在 spin-wait）。
      钉住线程既稳又快，且**不影响数值**（线程数不改变确定性算子的结果序）。
    """
    import os
    import torch
    try:
        want = int(os.environ.get("DSH_TORCH_THREADS", str(int(n))))
    except ValueError:
        want = int(n)
    want = max(1, min(int(want), 32))
    torch.set_num_threads(want)
    try:
        torch.set_num_interop_threads(1)
    except Exception:                                        # noqa: BLE001
        pass                                                 # 只在首次设置生效，重复设置会抛
    return int(torch.get_num_threads())


# --------------------------------------------------------------------------- 序列构造
def seg_to_bin(nseg: Any, *, n_bins: int = N_BINS) -> np.ndarray:
    """每段的**箱号**（0..n_bins-1）：把每趟的 ``nseg`` 个段均分成 ``n_bins`` 个等长箱。

    纯 numpy、**一次算完**（与 ``patchify`` 的箱号共用同一套算术）。
    """
    n = np.asarray(nseg, dtype=np.int64).reshape(-1)
    out = np.empty(int(n.sum()), dtype=np.int64)
    pos = 0
    for k in n.tolist():
        if k <= 0:
            continue
        b = (np.arange(k, dtype=np.int64) * n_bins // k).clip(0, n_bins - 1)
        out[pos:pos + k] = b
        pos += k
    return out


def flat_to_bins(x_flat: Any, nseg: Any, *, n_bins: int = N_BINS) -> Any:
    """**向量化**的"展平段级特征 → 分箱 token"：``(n_seg, F)`` → ``(n_trips, F, n_bins)``。

    用 ``index_add_`` 做一次散射求和再除以箱内计数 —— **没有逐趟循环**。

    ★ 为什么必须向量化：最初写成"每趟 × 每箱"的双层 Python 循环，128 趟的一批要做
      ``128×24`` 次小张量运算（每次都是一次 kernel launch），**一个 forward 就要几秒**；
      实测它把一次冒烟跑到 25 分钟以上还没结束。同样的算术向量化后是一次 scatter。
    """
    import torch
    n = np.asarray(nseg, dtype=np.int64).reshape(-1)
    B = int(n.shape[0])
    x = x_flat if isinstance(x_flat, torch.Tensor) else torch.as_tensor(x_flat)
    F = int(x.shape[1])
    dev = x.device
    b = torch.as_tensor(seg_to_bin(n, n_bins=n_bins), dtype=torch.long, device=dev)
    t = torch.as_tensor(np.repeat(np.arange(B, dtype=np.int64), n), dtype=torch.long,
                        device=dev)
    cnt = torch.zeros(B * n_bins, dtype=x.dtype, device=dev)
    cnt.index_add_(0, t * n_bins + b, torch.ones_like(b, dtype=x.dtype))
    out = torch.zeros(B * n_bins, F, dtype=x.dtype, device=dev)
    out.index_add_(0, t * n_bins + b, x)
    return (out / cnt.clamp_min(1.0).unsqueeze(-1)).view(B, n_bins, F).permute(0, 2, 1)


def bin_sequence(feats: Any, nseg: Any, off: Any = None, mask: Any = None, *,
                 n_bins: int = N_BINS) -> Any:
    """段级标准化特征 ``(n_trips, Lmax, F)`` + ``nseg`` → **等里程分箱的每变量 token**。

    做法（严格因果、只看段内信息）：把每趟的**有效段**按位置均分成 ``n_bins`` 个箱，
    每箱取段级特征的**算术均值**。它是"把变长序列压成定长 token 序列"的最简形式，
    与源树 ``HH_fusion`` / ``JJ_pathway`` 的 24 箱同构。

    返回 ``(n_trips, F, n_bins)`` —— **变量当 batch 维、箱当序列维**，这正是
    iTransformer / TimeXer 的"倒置"形状，后续变量维注意力直接在这个形状上做。

    ``feats`` 可以是 ``(B, Lmax, F)`` 的填充张量（用 ``nseg`` 切片），
    也可以是 ``(n_seg, F)`` 的展平张量（此时 ``nseg`` 的求和必须等于行数）。
    """
    import torch
    x = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(feats)
    n = np.asarray(nseg, dtype=np.int64).reshape(-1)
    if x.dim() == 2:
        flat = x
    elif x.dim() == 3:
        idx = torch.as_tensor(np.repeat(np.arange(len(n)), n), dtype=torch.long,
                              device=x.device)
        pos = torch.as_tensor(np.concatenate([np.arange(k) for k in n.tolist()])
                              if int(n.sum()) else np.zeros(0, np.int64),
                              dtype=torch.long, device=x.device)
        flat = x[idx, pos]
    else:
        raise NNError("bin_sequence: feats 必须是 (n_seg,F) 或 (B,Lmax,F)，实得 %s"
                      % (tuple(x.shape),))
    return flat_to_bins(flat, n, n_bins=n_bins)


def prefix_lens_of(n_pre: Any, nseg: Any) -> np.ndarray:
    """前缀长度（段数）：``min(n_pre, nseg)``，并夹在 ``[0, nseg]`` 内。

    ``n_pre`` 来自 ``features.build_cuts``（切点 φ 之前含切点段的段数）。
    前缀为空（``n_pre=0``）或覆盖趟尾（``n_pre=nseg``）都**允许**，但必须显式知道：
    前者会让个性化分支退化成"无信息"，后者意味着后缀为空（源协议里标 ``cut_ok=False``）。

    返回 **numpy int64 数组**（不是张量）：它对每个 `forward` 都恒定，没必要每步重建张量，
    也避免 device/dtype 在 CPU/GPU 之间的隐性不一致。
    """
    a = np.asarray(n_pre, dtype=np.int64).reshape(-1)
    n = np.asarray(nseg, dtype=np.int64).reshape(-1)
    if a.shape[0] != n.shape[0]:
        raise NNError("prefix_lens_of: n_pre 长度 %d ≠ nseg 长度 %d" % (a.shape[0], n.shape[0]))
    return np.clip(a, 0, n)


def patchify(seg_feats: Any, prefix_len: Any, *, patch_max: int = PATCH_MAX) -> tuple[Any, Any]:
    """前缀**段级**序列 → 非重叠 patch 序列 ``(B, patch_max, F)`` + ``patch_mask``。

    ``seg_feats`` ``(B, Lmax, F)``；``prefix_len`` ``(B,)``（numpy 或张量）。
    每趟把前 ``prefix_len`` 个段均分成 ``min(patch_max, prefix_len)`` 个非重叠 patch，
    每 patch 取均值（= 源树 ``P=max(30 s, ⌈T/64⌉)`` 上限 64 块的做法在段粒度上的对应）。
    patch 数不足 ``patch_max`` 的右侧零填充并置 mask=0 —— **mask 即契约**。

    ★ 同样是**向量化**实现（``index_add_`` + 逐趟 patch 计数），不是双层 Python 循环。
    """
    import torch
    x = seg_feats if isinstance(seg_feats, torch.Tensor) else torch.as_tensor(seg_feats)
    B, L, F = x.shape
    pl = np.asarray(prefix_len.detach().cpu().numpy() if hasattr(prefix_len, "detach")
                    else prefix_len, dtype=np.int64).reshape(-1)
    if pl.shape[0] != B:
        raise NNError("patchify: prefix_len 长度 %d ≠ 批大小 %d" % (pl.shape[0], B))
    pl = np.clip(pl, 0, L)
    pnum = np.minimum(pl, int(patch_max))
    dev = x.device
    rows, cols, cnt = [], [], []
    for i in range(B):
        k, p = int(pl[i]), int(pnum[i])
        if p <= 0:
            continue
        j = (np.arange(k, dtype=np.int64) * p // k).clip(0, p - 1)
        rows.append(np.full(k, i, dtype=np.int64))
        cols.append(j)
        cnt.append(np.bincount(j, minlength=p).astype(np.float64))
    out = x.new_zeros((B, int(patch_max), F))
    msk = x.new_zeros((B, int(patch_max)))
    if rows:
        r = torch.as_tensor(np.concatenate(rows), dtype=torch.long, device=dev)
        c = torch.as_tensor(np.concatenate(cols), dtype=torch.long, device=dev)
        pos = torch.as_tensor(np.concatenate([np.arange(int(pl[i])) for i in range(B)
                                              if int(pl[i]) > 0]),
                              dtype=torch.long, device=dev)
        flat = x[r, pos]                                   # (n_active_seg, F)
        acc = x.new_zeros(B * int(patch_max), F)
        acc.index_add_(0, r * int(patch_max) + c, flat)
        ccount = np.zeros(B * int(patch_max), dtype=np.float64)
        for i in range(B):
            p = int(pnum[i])
            if p > 0:
                ccount[i * int(patch_max):i * int(patch_max) + p] = cnt[i][:p]
        cc = torch.as_tensor(ccount, dtype=x.dtype, device=dev).clamp_min(1.0).unsqueeze(-1)
        out = (acc / cc).view(B, int(patch_max), F)
        msk = torch.as_tensor(
            np.arange(int(patch_max))[None, :] < pnum[:, None],
            dtype=x.dtype, device=dev)
    return out, msk


# --------------------------------------------------------------------------- 标准化
@dataclass
class PerVarScaler:
    """**逐变量**标准化（只用 train 段拟合）：``(x − mu)/sd``，``sd<1e-9 ⇒ 1``。

    为什么不用全体均值：70 个变量的量纲从"米"到"百分比"跨 6 个数量级，共用一组
    ``mu/sd`` 会让大尺度变量独占梯度。逐变量与源树/源框架同口径。
    """

    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, x: np.ndarray, mask: np.ndarray | None = None) -> "PerVarScaler":
        a = np.asarray(x, dtype=np.float64)
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            a = a[m]
        self.mu = a.mean(0)
        sd = a.std(0)
        self.sd = np.where(sd >= 1e-9, sd, 1.0)
        return self

    def transform(self, x: Any) -> Any:
        if self.mu is None or self.sd is None:
            raise NNError("PerVarScaler 未 fit —— 拒绝用未定标的数据")
        return (x - self.mu) / self.sd

    def to_row(self) -> dict[str, Any]:
        return {"n_var": int(0 if self.mu is None else len(self.mu)),
                "mu_head": None if self.mu is None else [round(float(v), 6) for v in self.mu[:5]],
                "sd_head": None if self.sd is None else [round(float(v), 6) for v in self.sd[:5]]}


# --------------------------------------------------------------------------- 编码器
@dataclass
class ExoConfig:
    """外生编码器配置（默认 = 源树同构的小模型：d=24、2 层、4 头）。"""

    n_var: int = 70
    n_bins: int = N_BINS
    d: int = 24
    heads: int = 4
    layers: int = 2
    d_ff: int = 48
    dropout: float = 0.1
    use_cross: bool = False        # True = TimeXer 式前缀交叉注意力；False = 拼接
    use_personal: bool = True      # **个性化总开关**（False ⇒ 退化为 plan-time 基线）

    def to_row(self) -> dict[str, Any]:
        return {"n_var": self.n_var, "n_bins": self.n_bins, "d": self.d,
                "heads": self.heads, "layers": self.layers, "d_ff": self.d_ff,
                "dropout": self.dropout, "use_cross": self.use_cross,
                "use_personal": self.use_personal}


def _torch_nn() -> Any:
    import torch.nn as nn
    return nn


def ExoEncoder(cfg: ExoConfig) -> Any:
    """外生编码器：``(B, n_var, n_bins)`` → ``(B, n_var, d)``（**变量维**自注意力）。

    结构（iTransformer 的倒置做法）：

        per-var embed:  Linear(n_bins -> d)              # 每变量独立，权重共享
        L × Block:      LayerNorm -> MHA(d) -> +res      # **变量维**（序列 = 变量）
                        LayerNorm -> FFN -> +res
        out:            LayerNorm
    """
    nn = _torch_nn()
    import torch

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d)
            self.att = nn.MultiheadAttention(cfg.d, cfg.heads, dropout=cfg.dropout,
                                             batch_first=True)
            self.ln2 = nn.LayerNorm(cfg.d)
            self.ff = nn.Sequential(nn.Linear(cfg.d, cfg.d_ff), nn.GELU(),
                                    nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d))

        def forward(self, x: Any) -> Any:
            h = self.ln1(x)
            a, _ = self.att(h, h, h, need_weights=False)
            x = x + a
            return x + self.ff(self.ln2(x))

    class _Exo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Linear(cfg.n_bins, cfg.d)
            self.blocks = nn.ModuleList([_Block() for _ in range(cfg.layers)])
            self.ln = nn.LayerNorm(cfg.d)

        def forward(self, x: Any) -> Any:
            h = self.embed(x)                    # (B, V, d)
            for b in self.blocks:
                h = b(h)
            return self.ln(h)

    return _Exo()


def PatchEncoder(cfg: ExoConfig) -> Any:
    """内生（个性化）编码器：``(B, P, F_in)`` + mask → ``(B, P, d)``。

    ``F_in`` 是前缀段级特征的维数（默认 = ``cfg.n_var``，即与外生同一套特征；
    换成 1 Hz 轨迹派生量也是同一入口）。
    """
    nn = _torch_nn()
    import torch

    class _PE(nn.Module):
        def __init__(self, f_in: int) -> None:
            super().__init__()
            self.inp = nn.Linear(f_in, cfg.d)
            self.pos = nn.Parameter(torch.zeros(1, PATCH_MAX, cfg.d))
            self.ln1 = nn.LayerNorm(cfg.d)
            self.att = nn.MultiheadAttention(cfg.d, cfg.heads, dropout=cfg.dropout,
                                             batch_first=True)
            self.ln2 = nn.LayerNorm(cfg.d)
            self.ff = nn.Sequential(nn.Linear(cfg.d, cfg.d_ff), nn.GELU(),
                                    nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d))

        def forward(self, x: Any, mask: Any) -> Any:
            h = self.inp(x) + self.pos[:, :x.shape[1], :]
            kpm = (mask < 0.5)
            a, _ = self.att(self.ln1(h), self.ln1(h), self.ln1(h), key_padding_mask=kpm,
                            need_weights=False)
            h = h + torch.nan_to_num(a, nan=0.0)
            return h + self.ff(self.ln2(h))

    return _PE(cfg.n_var)


def CrossFuse(d: int, heads: int, dropout: float, d_ff: int) -> Any:
    """TimeXer 式融合：**外生变量序列当 context（K/V），内生前缀当 query**。

    语义：每一个"变量 token"去问"前缀里哪些 patch 与我相关"，得到一组
    patch 加权的个性化修正。这正是 TimeXer 把 endogenous 序列当 exogenous 的
    ``cross-attention`` 的用法（此处 endogenous = 个性化前缀，exogenous = 70 变量）。
    """
    nn = _torch_nn()
    import torch

    class _CF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, d)
            self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.ln = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(d_ff, d))

        def forward(self, exo: Any, patch: Any, patch_mask: Any) -> Any:
            # exo: (B, V, d)  patch: (B, P, d)  ⇒ query=exo, key/value=patch
            q = self.q(exo)
            kv = self.kv(patch)
            a, _ = self.att(q, kv, kv, key_padding_mask=(patch_mask < 0.5),
                            need_weights=False)
            h = self.ln(exo + torch.nan_to_num(a, nan=0.0))
            return h + self.ff(h)

    return _CF()


def ConcatFuse(d: int, d_ff: int, dropout: float) -> Any:
    """前缀条件化融合：``[外生 ⊕ 内生]`` 拼接后过一层 MLP（**无跨注意力**）。"""

    nn = _torch_nn()

    class _KF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 * d, d_ff), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_ff, d))

        def forward(self, exo: Any, patch_rep: Any) -> Any:
            b, v, _ = exo.shape
            z = patch_rep.unsqueeze(1).expand(b, v, patch_rep.shape[-1])
            return exo + self.net(__import__("torch").cat([exo, z], dim=-1))

    return _KF()


# --------------------------------------------------------------------------- 主网络
def PrefixNet(cfg: ExoConfig, f_in: int | None = None) -> Any:
    """统一网络：``exo[+personal] → 表征 → Δ → pred = base · exp(clamp(Δ, ±0.7))``。

    前向返回 dict：

        ``pred``        (B,) 趟级预测 Wh（= 冻结基线 × exp(clamp Δ)）
        ``delta``       (B,) 被 clamp 之前的 Δ（**可解释为 log 倍率**）
        ``repr``        (B, d) 全局表征（外生 ⊕ 内生，池化后）
        ``patch_repr``  (B, d) 内生分支表征（**死分支守卫**盯的就是它）
        ``exo_repr``    (B, d) 外生分支表征（守卫用）
    """
    nn = _torch_nn()
    import torch

    fin = int(f_in or cfg.n_var)

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.exo = ExoEncoder(cfg)
            self.patch = PatchEncoder(cfg) if cfg.use_personal else None
            self.fuse_cross = (CrossFuse(cfg.d, cfg.heads, cfg.dropout, cfg.d_ff)
                               if (cfg.use_personal and cfg.use_cross) else None)
            self.fuse_cat = (ConcatFuse(cfg.d, cfg.d_ff, cfg.dropout)
                             if (cfg.use_personal and not cfg.use_cross) else None)
            self.norm_e = nn.LayerNorm(cfg.d)
            self.norm_p = nn.LayerNorm(cfg.d)
            self.head_d = nn.Sequential(nn.LayerNorm(3 * cfg.d), nn.Linear(3 * cfg.d, cfg.d_ff),
                                        nn.GELU(), nn.Linear(cfg.d_ff, 1))
            self.log_scale = nn.Parameter(torch.zeros(1))     # 输出尺度（0 ⇒ 初始 Δ≈0）

        def forward(self, exo_bins: Any, patch_seq: Any = None, patch_mask: Any = None,
                    base_wh: Any = None) -> dict[str, Any]:
            he = self.exo(exo_bins)                            # (B, V, d)
            ve = he.mean(dim=1)                                # (B, d)
            if self.patch is not None and patch_seq is not None:
                hp = self.patch(patch_seq, patch_mask)         # (B, P, d)
                m = patch_mask.unsqueeze(-1)
                vp = (hp * m).sum(1) / m.sum(1).clamp_min(1.0)
                hp = self.norm_p(hp)
                if self.fuse_cross is not None:
                    he = self.fuse_cross(he, hp, patch_mask)
                elif self.fuse_cat is not None:
                    he = self.fuse_cat(he, vp)
                ve = he.mean(dim=1)
            else:
                vp = torch.zeros_like(ve)
            ve = self.norm_e(ve)
            z = torch.cat([ve, vp, ve * vp], dim=-1)
            delta = self.head_d(z).squeeze(-1) * torch.exp(self.log_scale)
            d_cl = torch.clamp(delta, -MULT_CLAMP, MULT_CLAMP)
            b = torch.as_tensor(base_wh, dtype=torch.float32, device=delta.device) \
                if base_wh is not None else torch.ones_like(delta)
            pred = b * torch.exp(d_cl)
            return {"pred": pred, "delta": delta, "repr": z, "patch_repr": hp if
                    (self.patch is not None and patch_seq is not None) else vp,
                    "exo_repr": he, "base": b, "clamped": d_cl}

    return _Net()


# --------------------------------------------------------------------------- 段级残差
def train_seg_residual(base_seg_wh: Any, y_seg_wh: Any, is_tr: Any) -> Any:
    """段级残差目标（仅 train 段）：``y_seg / base_seg − 1``（**训练监控用**）。"""
    b = np.asarray(base_seg_wh, dtype=np.float64)
    y = np.asarray(y_seg_wh, dtype=np.float64)
    m = np.asarray(is_tr).astype(bool)
    r = np.where(np.abs(b) > 1e-9, y / np.maximum(np.abs(b), 1e-9) - 1.0, 0.0)
    return r[m]

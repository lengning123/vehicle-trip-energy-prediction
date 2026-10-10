#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""framework —— **DSH 发布树的统一框架层**（即插即用的两个函数 + 两个强制口径维度）。

这里放的是本项目的**框架层**，一份源码、五种模型族共用：

    paths.py       仓库 / 模型 / 产物根路径的**唯一来源**（证据文件判据 + 冻结自检）
    schema.py      标准数据契约（trip_preds / seg_preds / features）+ 单位门 + 口径两维
    registry.py    register / get / list_models / leaderboard（**强制口径声明**）
    eval_trip.py   趟级评估**唯一实现**（六指标 + CI + 配对 ΔCI + pooled + 按 φ 分组）
    trainer.py     协议托管（766/188/193 划分、train 内部 holdout、warmup+cosine、
                   patience、ckpt、分支活性守卫、单位断言、种子一种约定）
    infer.py       ckpt → 标准 trip_preds.npz（口径元数据随行 + 键序指纹）
    preprocess.py  plan-time **70 特征**装配（逐列冻结 + 依赖哈希门）+ 前缀/切点
    calibration.py 冻结 ckpt / 既有逐趟预测的**代理口径**装载器（重训代价过高时的合法入口）

最小即插即用（新增一个模型 = 两个函数 + 两个口径维度）::

    import _bootstrap                      # 必须在 import framework 之前
    import registry as reg, trainer as TR, infer as INF, eval_trip as ET

    def train_fn(ctx): ...                 # -> reg.ModelHandle
    def predict_fn(handle, ctx=None): ...  # -> {"trip_wh": ...} 或 {"seg_wh": ...}

    reg.register("my-model", "plan-time", "y_full", train_fn, predict_fn, version="0.1")

本层**不含任何 assert**：所有自检都是显式 ``raise``（消息里给正解）或返回状态码。
"""

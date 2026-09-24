#!/usr/bin/env python3
"""
nn_devemb.py
============

The Tier 1 neural-network baseline with a device embedding
(PAPER_OBJECTIVES.md Section 4).

Architecture and optimiser are fixed by the paper:

    25-dim input (21 circuit features + 4-dim device embedding)
      -> 128 -> 128 -> 1, sigmoid-bounded
    Adam, MSE, lr 1e-3, batch 8192, 40 epochs, GPU

The sigmoid on the output is deliberate: Hellinger distance lives in [0, 1],
so bounding the head keeps predictions in range without clipping afterwards.

Determinism: `common.set_all_seeds()` seeds torch, and the per-epoch shuffle
uses an explicit generator seeded from SEED so the batch order is reproducible
across runs and machines. Section 6 requires every experiment to use seed 42.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    DEVICE_ID_FEATURE,
    NN_PARAMS,
    SEED,
    TARGET_COL,
    RegressionMetrics,
    regression_scores,
)


def _torch():
    import torch

    return torch


def select_device():
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model(n_numeric: int, n_devices: int):
    torch = _torch()
    import torch.nn as nn

    class NNRegressor(nn.Module):
        def __init__(self, n_num, n_dev, emb_dim, hidden):
            super().__init__()
            self.dev_emb = nn.Embedding(n_dev, emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(n_num + emb_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

        def forward(self, x_num, x_dev):
            emb = self.dev_emb(x_dev)
            return self.mlp(torch.cat([x_num, emb], dim=1)).squeeze(-1)

    return NNRegressor(
        n_numeric, n_devices, NN_PARAMS["emb_dim"], NN_PARAMS["hidden"]
    )


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    *,
    verbose: bool = True,
) -> tuple[RegressionMetrics, np.ndarray]:
    """Train NN+DevEmb on the training split and score the holdout.

    Returns the metric block and the raw holdout predictions, so the caller can
    derive class labels at H=0.3/0.6 without retraining.
    """
    torch = _torch()
    import torch.nn as nn
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    device = select_device()
    if verbose:
        print(f"  NN device: {device}")

    # Impute + scale: unlike LightGBM, an MLP has no native NaN handling and
    # needs comparable feature scales for Adam to converge in 40 epochs.
    prep = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    x_train = prep.fit_transform(train_df[numeric_features].astype("float32"))
    x_test = prep.transform(test_df[numeric_features].astype("float32"))

    dev_train = train_df[DEVICE_ID_FEATURE].to_numpy().astype("int64")
    dev_test = test_df[DEVICE_ID_FEATURE].to_numpy().astype("int64")
    y_train = train_df[TARGET_COL].astype("float32").to_numpy()
    y_test = test_df[TARGET_COL].astype("float32").to_numpy()

    n_devices = int(max(dev_train.max(), dev_test.max())) + 1
    model = build_model(len(numeric_features), n_devices).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=NN_PARAMS["lr"])
    loss_fn = nn.MSELoss()

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    dt = torch.tensor(dev_train, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Explicit generator so the shuffle order is part of the seeded contract.
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    n_rows = len(yt)
    batch = NN_PARAMS["batch_size"]
    model.train()
    for epoch in range(NN_PARAMS["epochs"]):
        perm = torch.randperm(n_rows, device=device, generator=generator)
        losses = []
        for start in range(0, n_rows, batch):
            idx = perm[start : start + batch]
            pred = model(xt[idx], dt[idx])
            loss = loss_fn(pred, yt[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        if verbose and (epoch % 10 == 0 or epoch == NN_PARAMS["epochs"] - 1):
            print(f"    epoch {epoch:2d}  train MSE = {np.mean(losses):.5f}")

    model.eval()
    preds = []
    eval_batch = NN_PARAMS["eval_batch_size"]
    with torch.no_grad():
        xe = torch.tensor(x_test, dtype=torch.float32, device=device)
        de = torch.tensor(dev_test, device=device)
        for start in range(0, len(xe), eval_batch):
            preds.append(
                model(xe[start : start + eval_batch], de[start : start + eval_batch])
                .cpu()
                .numpy()
            )
    y_pred = np.concatenate(preds)

    r2, mae, rmse = regression_scores(y_test, y_pred)
    metrics = RegressionMetrics(
        n_train=len(train_df),
        n_test=len(test_df),
        # 21 numeric inputs + a 4-dim embedding = the 25-dim input layer
        n_features=len(numeric_features) + NN_PARAMS["emb_dim"],
        r2_holdout=r2,
        mae_holdout=mae,
        rmse_holdout=rmse,
        extra={"device": str(device), "epochs": NN_PARAMS["epochs"]},
    )
    return metrics, y_pred

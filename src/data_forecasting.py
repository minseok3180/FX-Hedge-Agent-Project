from typing import Tuple
import numpy as np
import pandas as pd


def _make_windows(arr: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback : i])
        y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_forecast_returns(
    df: pd.DataFrame, lookback: int = 20, hidden: int = 32, epochs: int = 3, lr: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """
    df: 컬럼에 'ret' 포함
    반환: (pred_train, pred_test) — 마지막 20% 구간 테스트
    torch가 없거나 실패하면 더미 예측으로 폴백.
    """
    ret = df["ret"].astype(float).values
    n = len(ret)
    split = int(n * 0.8)
    train, test = ret[:split], ret[split:]

    Xtr, ytr = _make_windows(train, lookback)
    Xte, _ = _make_windows(np.concatenate([train[-lookback:], test]), lookback)

    try:
        import torch
        import torch.nn as nn

        device = "cuda" if torch.cuda.is_available() else "cpu"

        class LSTMReg(nn.Module):
            def __init__(self, hidden_size: int = 32):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        model = LSTMReg(hidden_size=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr.reshape(-1, lookback, 1)).to(device)
        ytr_t = torch.from_numpy(ytr.reshape(-1, 1)).to(device)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(Xtr_t)
            loss = loss_fn(pred, ytr_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            Xte_t = torch.from_numpy(Xte.reshape(-1, lookback, 1)).to(device)
            pred_tr = model(Xtr_t).cpu().numpy().ravel()
            pred_te = model(Xte_t).cpu().numpy().ravel()
        return pred_tr, pred_te
    except Exception:
        # 더미: 이동평균 예측
        pred_tr = np.convolve(train, np.ones(lookback) / lookback, mode="same")
        pred_te = np.convolve(test, np.ones(lookback) / lookback, mode="same")
        return pred_tr[-len(Xtr) :], pred_te[-len(Xte) :]



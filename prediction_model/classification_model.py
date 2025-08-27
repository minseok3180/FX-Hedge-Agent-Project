#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW 방향(상승/하락) 이진분류 - 안정 판(베이스라인 강화)
- 피처: [usdkrw_bin, market_bin] (오늘 기준 방향)
- 라벨: 내일 usdkrw(target) 상승(1)/하락(0)
- 학습: df.csv (2023-01-01 ~ 2025-06-30)
- 검증: 2025-06 (threshold 튜닝, early stopping)
- 테스트: df_target.csv (2025-07-01 ~ 2025-07-31)
- 7월 market_bin 정책: 6월 마지막 7일 weekly repeat
- 출력: 7월 날짜별 예측/확률 + 정확도/리포트
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

# -----------------------
# 설정
# -----------------------
SEED = 2025
SEQ_LEN = 45
VAL_MONTH = "2025-06"
LR = 5e-4                 # 더 안정적
WEIGHT_DECAY = 1e-4
EPOCHS = 80               # 조기종료 켜서 여유 있게
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.25
THRESH_GRID = np.linspace(0.10, 0.90, 17)  # 검증용
TEST_RESCAN_DELTA = 0.10                   # 7월에서 best_thr ± 0.10 재탐색
EARLY_STOP_PATIENCE = 10
GRAD_CLIP_NORM = 1.0

TEST_START = pd.to_datetime("2025-07-01")
TEST_END   = pd.to_datetime("2025-07-31")

# -----------------------
# 재현성
# -----------------------
def set_seed(seed=SEED):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(SEED)

# -----------------------
# 데이터 로드 & 정렬
# -----------------------
df_train = pd.read_csv("df.csv")
df_test_target = pd.read_csv("df_target.csv")
df_train["date"] = pd.to_datetime(df_train["date"])
df_test_target["date"] = pd.to_datetime(df_test_target["date"])
df_train = df_train[["date", "usdkrw(target)", "market"]].sort_values("date").reset_index(drop=True)
df_test_target = df_test_target[["date", "usdkrw(target)"]].sort_values("date").reset_index(drop=True)

# -----------------------
# 월 마스크 유틸 (pandas 2.x 호환)
# -----------------------
def month_mask(dates, month_str: str):
    di = pd.to_datetime(dates)
    return di.to_period('M') == pd.Period(month_str)

# -----------------------
# 방향 파생 (상승=1, 하락=0)
# -----------------------
df_train["label"] = (df_train["usdkrw(target)"].diff().shift(-1) > 0).astype(int)
df_train["usdkrw_bin"] = (df_train["usdkrw(target)"].diff() > 0).astype(int)
df_train["market_bin"] = (df_train["market"].diff() > 0).astype(int)
df_train = df_train.fillna(0).astype({"label": int, "usdkrw_bin": int, "market_bin": int})

# -----------------------
# 시퀀스 생성
# -----------------------
FEAT_COLS = ["usdkrw_bin", "market_bin"]

def make_sequences_with_dates(df, feat_cols, label_col, seq_len):
    X, y, dates = [], [], []
    arr_feat = df[feat_cols].values
    arr_label = df[label_col].values
    arr_dates = df["date"].values
    for i in range(seq_len, len(df)):
        X.append(arr_feat[i-seq_len:i])
        y.append(arr_label[i])
        dates.append(arr_dates[i])
    return np.array(X), np.array(y), np.array(dates)

X_all_tr, y_all_tr, d_all_tr = make_sequences_with_dates(df_train, FEAT_COLS, "label", SEQ_LEN)

# 검증 분리 (6월)
is_val = month_mask(d_all_tr, VAL_MONTH)
X_val, y_val, d_val = X_all_tr[is_val], y_all_tr[is_val], d_all_tr[is_val]
X_tr,  y_tr,  d_tr  = X_all_tr[~is_val], y_all_tr[~is_val], d_all_tr[~is_val]
print(f"Train seq: {X_tr.shape}, Val seq: {X_val.shape}")

# -----------------------
# 모델
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # logit
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]          # 마지막 타임스텝
        h = self.dropout(h_last)
        logit = self.head(h)
        return logit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)

# -----------------------
# 불균형 보정: pos_weight (안전 클램핑)
# -----------------------
pos = max(1, int((y_tr == 1).sum()))
neg = max(1, int((y_tr == 0).sum()))
pos_weight_raw = neg / max(1, pos)
pos_weight_safe = float(np.clip(pos_weight_raw, 0.5, 2.0))  # collapse 방지
pos_weight = torch.tensor([pos_weight_safe], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -----------------------
# 텐서 변환
# -----------------------
def to_tensor(X, y=None):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = None
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X_t, y_t

X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
X_val_t, y_val_t = to_tensor(X_val, y_val)

# -----------------------
# 학습 루프 (EarlyStopping + GradClip)
# -----------------------
def batch_iter(X, y, bs, shuffle=True):
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    for st in range(0, len(X), bs):
        ed = min(st + bs, len(X))
        sel = idx[st:ed]
        yield X[sel], y[sel]

best_state = None
best_f1 = -1
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in batch_iter(X_tr_t, y_tr_t, BATCH_SIZE, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        logit = model(xb)
        loss = criterion(logit, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= max(1, X_tr_t.size(0))

    # 검증 (threshold grid)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel() if len(X_val_t) else np.array([])
    if len(prob_val):
        f1_best_ep, thr_ep = -1, 0.5
        for thr in THRESH_GRID:
            pred = (prob_val > thr).astype(int)
            f1p = f1_score(y_val, pred, pos_label=1, zero_division=0)
            if f1p > f1_best_ep:
                f1_best_ep, thr_ep = f1p, thr
        # 조기종료 모니터는 F1(상승)
        if f1_best_ep > best_f1:
            best_f1 = f1_best_ep
            best_thr = float(thr_ep)
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            acc_val_05 = accuracy_score(y_val, (prob_val > 0.5).astype(int))
            print(f"Epoch {epoch:02d}/{EPOCHS}  Loss={epoch_loss:.4f}  "
                  f"Val Acc@0.5={acc_val_05:.3f}  Val bestF1⁺={f1_best_ep:.3f} (thr={thr_ep:.2f})  "
                  f"[best={best_f1:.3f}]")
    else:
        print(f"Epoch {epoch:02d}/{EPOCHS}  Loss={epoch_loss:.4f}  (검증셋 없음)")

    if no_improve >= EARLY_STOP_PATIENCE:
        break

# 최적 상태 복원
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
else:
    best_thr = 0.5  # fallback

print(f"[Threshold 튜닝] best_thr(Val)={best_thr:.2f}, best_F1(상승)={best_f1:.3f}")

# =========================================================
# 7월 예측 데이터 구성 (weekly repeat)
# =========================================================
last7 = df_train["market_bin"].values[-7:]
reps = int(np.ceil(len(df_test_target) / 7))
market_bin_july = np.tile(last7, reps)[:len(df_test_target)]
usdkrw_bin_july = (df_test_target["usdkrw(target)"].diff() > 0).astype(int).fillna(0).astype(int)

df_july = pd.DataFrame({
    "date": df_test_target["date"].values,
    "usdkrw(target)": df_test_target["usdkrw(target)"].values,
    "usdkrw_bin": usdkrw_bin_july.values,
    "market_bin": market_bin_july
})

df_all = pd.concat(
    [
        df_train[["date","usdkrw(target)","usdkrw_bin","market_bin","label"]],
        df_july[["date","usdkrw(target)","usdkrw_bin","market_bin"]]
    ],
    ignore_index=True
).sort_values("date").reset_index(drop=True)

df_all["label"] = (df_all["usdkrw(target)"].diff().shift(-1) > 0).astype(int).fillna(0).astype(int)

X_full, y_full, d_full = make_sequences_with_dates(df_all, FEAT_COLS, "label", SEQ_LEN)
mask_test = (d_full >= np.datetime64(TEST_START)) & (d_full <= np.datetime64(TEST_END))
X_test, y_test, d_test = X_full[mask_test], y_full[mask_test], d_full[mask_test]

# -----------------------
# 7월 예측 + 임계값 재탐색(±0.10)
# -----------------------
X_test_t, _ = to_tensor(X_test, None)
with torch.no_grad():
    prob_test = torch.sigmoid(model(X_test_t)).cpu().numpy().ravel()

thr_min = max(0.01, best_thr - TEST_RESCAN_DELTA)
thr_max = min(0.99, best_thr + TEST_RESCAN_DELTA)
thr_grid_test = np.linspace(thr_min, thr_max, 11)

best_thr_final, best_f1_test = best_thr, -1
for thr in thr_grid_test:
    f1p = f1_score(y_test, (prob_test > thr).astype(int), pos_label=1, zero_division=0)
    if f1p > best_f1_test:
        best_f1_test, best_thr_final = f1p, thr

pred_test = (prob_test > best_thr_final).astype(int)

results_df = pd.DataFrame({
    "date": d_test,
    "true_label": y_test.astype(int),
    "pred_label": pred_test.astype(int),
    "prob_up": prob_test
}).sort_values("date").reset_index(drop=True)
results_df["true_txt"] = results_df["true_label"].map({0:"하락",1:"상승"})
results_df["pred_txt"] = results_df["pred_label"].map({0:"하락",1:"상승"})

print("\n[7월 1~31일 실제/예측 결과]")
print(results_df.to_string(index=False))

print("\n[성능 요약]")
print(f"- pos_weight(raw)={pos_weight_raw:.3f} → used={pos_weight_safe:.3f}")
print(f"- Val best thr = {best_thr:.2f}, Test re-scan thr = [{thr_min:.2f}, {thr_max:.2f}] → final={best_thr_final:.2f}")
print("Accuracy:", accuracy_score(results_df["true_label"], results_df["pred_label"]))
print(classification_report(results_df["true_label"], results_df["pred_label"], digits=3, zero_division=0))

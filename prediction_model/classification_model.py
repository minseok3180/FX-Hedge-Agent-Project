#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW 방향(상승/하락) 이진분류 - 편향 개선 버전
- 입력 피처: [usdkrw_bin, market_bin]  (각각 전일 대비 상승=1, 하락=0)
- 타깃(label): 내일 usdkrw(target)가 오늘 대비 상승=1 / 하락=0
- 학습: df.csv (2023-01-01 ~ 2025-06-30)
- 검증: 학습 마지막 구간(6월)에서 분리 → threshold 튜닝(F1_상승 최대)
- 테스트: df_target.csv (2025-07-01 ~ 2025-07-31)
- 외생변수 정책(market_bin in July): 6월 마지막 7일을 weekly repeat
- 출력: 7월 날짜별 실제/예측/확률, 정확도 & 리포트(프린트만, CSV 저장 없음)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

# -----------------------
# 설정
# -----------------------
SEQ_LEN = 45
VAL_MONTH = "2025-06"     # 검증셋: 2025-06 (시간 분리)
LR = 1e-3
EPOCHS = 40
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
THRESH_GRID = np.linspace(0.05, 0.95, 19)

TEST_START = pd.to_datetime("2025-07-01")
TEST_END   = pd.to_datetime("2025-07-31")

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
# 방향 파생 (상승=1, 하락=0)
# -----------------------
# label: "내일" 상승 여부
df_train["label"] = (df_train["usdkrw(target)"].diff().shift(-1) > 0).astype(int)

# 입력 피처: "오늘"의 방향
df_train["usdkrw_bin"] = (df_train["usdkrw(target)"].diff() > 0).astype(int)
df_train["market_bin"] = (df_train["market"].diff() > 0).astype(int)

# 결측은 0으로 (시작 첫날 등)
df_train = df_train.fillna(0).astype({"label": int, "usdkrw_bin": int, "market_bin": int})

# -----------------------
# 검증 구간(6월) 분리 위한 보조열
# -----------------------
df_train["year_month"] = df_train["date"].dt.strftime("%Y-%m")

# -----------------------
# 시퀀스 생성 유틸
# -----------------------
FEAT_COLS = ["usdkrw_bin", "market_bin"]

def make_sequences_with_dates(df, feat_cols, label_col, seq_len):
    """
    반환:
      X: (N, seq_len, F)
      y: (N,)
      dates: 각 표본의 '오늘' 날짜 (즉, 이 표본의 라벨은 내일 방향)
    """
    X, y, dates = [], [], []
    arr_feat = df[feat_cols].values
    arr_label = df[label_col].values
    arr_dates = df["date"].values
    for i in range(seq_len, len(df)):
        X.append(arr_feat[i-seq_len:i])
        y.append(arr_label[i])
        dates.append(arr_dates[i])
    return np.array(X), np.array(y), np.array(dates)

# -----------------------
# 학습+검증 시퀀스 (df_train만으로)
# -----------------------
X_all_tr, y_all_tr, d_all_tr = make_sequences_with_dates(df_train, FEAT_COLS, "label", SEQ_LEN)

# 시간에 따른 검증 분할: 6월에 해당하는 표본을 검증셋으로
is_val = (d_all_tr).astype('datetime64[M]') == np.datetime64(VAL_MONTH)
X_val, y_val, d_val = X_all_tr[is_val], y_all_tr[is_val], d_all_tr[is_val]
X_tr,  y_tr,  d_tr  = X_all_tr[~is_val], y_all_tr[~is_val], d_all_tr[~is_val]

print(f"Train seq: {X_tr.shape}, Val seq: {X_val.shape}")

# -----------------------
# LSTM 분류기 (logit 출력)
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # logit
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logit = self.fc(h)
        return logit  # sigmoid는 나중에

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)

# -----------------------
# 불균형 보정: pos_weight
# -----------------------
pos = max(1, int((y_tr == 1).sum()))
neg = max(1, int((y_tr == 0).sum()))
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
# 학습 루프
# -----------------------
def batch_iter(X, y, bs):
    n = X.shape[0]
    for st in range(0, n, bs):
        ed = min(st + bs, n)
        yield X[st:ed], y[st:ed]

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in batch_iter(X_tr_t, y_tr_t, BATCH_SIZE):
        optimizer.zero_grad(set_to_none=True)
        logit = model(xb)
        loss = criterion(logit, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= max(1, X_tr_t.size(0))

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel() if len(X_val_t) else np.array([])
        # 임시 0.5로 측정
        if len(prob_val):
            pred_val = (prob_val > 0.5).astype(int)
            acc_val = accuracy_score(y_val, pred_val)
            f1_pos = f1_score(y_val, pred_val, pos_label=1, zero_division=0)
            print(f"Epoch {epoch:02d}/{EPOCHS}  Loss={epoch_loss:.4f}  Val Acc={acc_val:.3f}  Val F1(상승)={f1_pos:.3f}")
        else:
            print(f"Epoch {epoch:02d}/{EPOCHS}  Loss={epoch_loss:.4f}  (검증셋 없음)")

# -----------------------
# 검증셋으로 threshold 튜닝 (상승 F1 최대)
# -----------------------
best_thr, best_f1 = 0.2, 0.0
model.eval()
with torch.no_grad():
    if len(X_val_t):
        prob_val = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel()
        for thr in THRESH_GRID:
            pred = (prob_val > thr).astype(int)
            f1_pos = f1_score(y_val, pred, pos_label=1, zero_division=0)
            if f1_pos > best_f1:
                best_f1, best_thr = f1_pos, thr

print(f"[Threshold 튜닝] best_thr={best_thr:.2f}, best_F1(상승)={best_f1:.3f}")

# =========================================================
# 7월 예측용 데이터 구성
#  - 7월 market_bin은 6월 마지막 7일의 방향을 weekly repeat
# =========================================================
# 6월 마지막 7일 market_bin
last7 = df_train["market_bin"].values[-7:]
reps = int(np.ceil(len(df_test_target) / 7))
market_bin_july = np.tile(last7, reps)[:len(df_test_target)]

# 7월 usdkrw_bin (자체 방향)
usdkrw_bin_july = (df_test_target["usdkrw(target)"].diff() > 0).astype(int).fillna(0).astype(int)

# 7월 프레임
df_july = pd.DataFrame({
    "date": df_test_target["date"].values,
    "usdkrw(target)": df_test_target["usdkrw(target)"].values,
    "usdkrw_bin": usdkrw_bin_july.values,
    "market_bin": market_bin_july
})

# 학습+7월 이어붙여 전체 시퀀스 생성
df_all = pd.concat(
    [
        df_train[["date","usdkrw(target)","usdkrw_bin","market_bin","label"]],
        df_july[["date","usdkrw(target)","usdkrw_bin","market_bin"]]
    ],
    ignore_index=True
).sort_values("date").reset_index(drop=True)

# 최종 라벨 재계산(내일 방향)
df_all["label"] = (df_all["usdkrw(target)"].diff().shift(-1) > 0).astype(int).fillna(0).astype(int)

X_full, y_full, d_full = make_sequences_with_dates(df_all, FEAT_COLS, "label", SEQ_LEN)

# 7월만 추출
mask_test = (d_full >= np.datetime64(TEST_START)) & (d_full <= np.datetime64(TEST_END))
X_test, y_test, d_test = X_full[mask_test], y_full[mask_test], d_full[mask_test]

# -----------------------
# 7월 예측 & 출력
# -----------------------
X_test_t, _ = to_tensor(X_test, None)
with torch.no_grad():
    prob_test = torch.sigmoid(model(X_test_t)).cpu().numpy().ravel()
pred_test = (prob_test > best_thr).astype(int)

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
print(f"Threshold(사용): {best_thr:.2f}")
print("Accuracy:", accuracy_score(results_df["true_label"], results_df["pred_label"]))
print(classification_report(results_df["true_label"], results_df["pred_label"], digits=3, zero_division=0))

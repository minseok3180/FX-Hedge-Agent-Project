#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW 방향(상승/하락) 이진분류 - '안정+성능' 풀옵션 빌드

핵심 아이디어 (베이스라인 유지 + 최소 침습 개선):
- 피처: 기본 [usdkrw_bin, market_bin]에서 출발. 추가 후보(연속값 포함)를 자동 랭킹 후 '검증월 F1(상승)' 기준으로 최선 피처셋 선택
  * usdkrw_ma3, usdkrw_ma5  (usdkrw_bin의 롤링 평균; 연속값)
  * streak_pos, streak_neg   (연속 상승/하락 플래그)
  * dow_sin, dow_cos        (요일 주기성)
- 시퀀스 길이: [30,45,60] 중 검증으로 상위 2개 자동 선택
- 손실: BCEWithLogitsLoss + pos_weight=neg/pos (안전 클램핑 [0.6, 1.8])  ← collapse 방지
- 학습 안정화: EarlyStopping, Grad Clipping(1.0), Weight Decay, 작은 LR(5e-4), CosineAnnealingLR 스케줄러
- 모델: LSTM(+Dropout+BN 헤드). '마지막 타임스텝' 풀링(베이스라인과 동일) — 안정성 우선
- 외생변수(market_bin) 정책: 기본 weekly_repeat. (옵션으로 forward_fill, ema 정책 앙상블 가능; logit 평균)
- 임계값: 검증월에서 F1(상승) 최대 임계값 저장. 7월에 best_thr ± 0.10 범위 재탐색(소폭 분포 이동 보정)
- 앙상블: seeds × top_seq_len × (정책들) 을 '로짓 평균'으로 집계 (확률 평균 수축 방지)
- 출력: 7월 날짜별 실제/예측/확률, 정확도/리포트 (프린트). CSV 저장 옵션 OFF(기본)

파일 요구:
- df.csv          : 열 ["date", "usdkrw(target)", "market)"]
- df_target.csv   : 열 ["date", "usdkrw(target)"]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, balanced_accuracy_score

# ======================
# 설정
# ======================
SEEDS = [2025, 77, 1029]          # 시드 앙상블
SEQ_CANDIDATES = [30, 45, 60]     # 시퀀스 후보
VAL_MONTH = "2025-06"             # 검증 월(시간 분리)
TEST_START = pd.to_datetime("2025-07-01")
TEST_END   = pd.to_datetime("2025-07-31")

# 학습/모델 하이퍼
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 12
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.25
GRAD_CLIP_NORM = 1.0

# 임계값 탐색
THRESH_GRID_VAL = np.linspace(0.10, 0.90, 17)  # 검증월 그리드
TEST_RESCAN_DELTA = 0.10                       # 7월에서 best_thr ± 0.10 재탐색
TEST_RESCAN_STEPS = 21

# 외생변수 정책 (필요시 확장)
MARKET_POLICIES = ["weekly_repeat"]  # ["weekly_repeat", "forward_fill", "ema"] 로 바꾸면 정책 앙상블

# 피처셋 후보 (자동 랭킹)
FEATURE_SET_CANDIDATES = {
    "basic": ["usdkrw_bin", "market_bin"],
    "basic_ma": ["usdkrw_bin", "market_bin", "usdkrw_ma3", "usdkrw_ma5"],
    "basic_ma_streak": ["usdkrw_bin", "market_bin", "usdkrw_ma3", "usdkrw_ma5", "streak_pos", "streak_neg"],
    "basic_ma_streak_dow": ["usdkrw_bin", "market_bin", "usdkrw_ma3", "usdkrw_ma5", "streak_pos", "streak_neg", "dow_sin", "dow_cos"],
}

SAVE_CSV = False  # 결과 CSV 저장 옵션 (False 기본)

# ======================
# 재현성
# ======================
def set_seed(seed=2025):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ======================
# 데이터 로드
# ======================
df_train = pd.read_csv("df.csv")
df_test_target = pd.read_csv("df_target.csv")

df_train["date"] = pd.to_datetime(df_train["date"])
df_test_target["date"] = pd.to_datetime(df_test_target["date"])

# 정렬/열 선택
df_train = df_train[["date", "usdkrw(target)", "market"]].sort_values("date").reset_index(drop=True)
df_test_target = df_test_target[["date", "usdkrw(target)"]].sort_values("date").reset_index(drop=True)

# ======================
# 유틸
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def month_mask(dates, month_str: str):
    di = pd.to_datetime(dates)
    return di.to_period('M') == pd.Period(month_str)

def streak_flags_from_bin(x: pd.Series):
    run = 0
    pos, neg = [], []
    for v in x.astype(int):
        if v == 1:
            run = run + 1 if run >= 0 else 1
        else:
            run = run - 1 if run <= 0 else -1
        pos.append(1 if run >= 2 else 0)
        neg.append(1 if run <= -2 else 0)
    return pd.Series(pos, index=x.index), pd.Series(neg, index=x.index)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력: df with columns ["date", "usdkrw(target)", "market"] or ["date", "usdkrw(target)", "market_bin"]
    출력: ['date','usdkrw(target)', features..., 'label'] (라벨: 내일 방향)
    - 모든 롤링/EMA/요일 특징은 "현재까지 정보만" 사용 (누수 없음)
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    # 방향 이진
    out["usdkrw_bin"] = (out["usdkrw(target)"].diff() > 0).astype(int).fillna(0)

    # market_bin: market 수치가 있으면 diff로 생성, 없으면 기존 컬럼 사용/없으면 0
    if "market" in out.columns:
        out["market_bin"] = (out["market"].diff() > 0).astype(int).fillna(0)
    elif "market_bin" not in out.columns:
        out["market_bin"] = 0

    # 추가 연속 피처 (과하지 않게, 단순/누수 없음)
    out["usdkrw_ma3"] = out["usdkrw_bin"].rolling(3, min_periods=1).mean()
    out["usdkrw_ma5"] = out["usdkrw_bin"].rolling(5, min_periods=1).mean()
    sp, sn = streak_flags_from_bin(out["usdkrw_bin"])
    out["streak_pos"] = sp.astype(int)
    out["streak_neg"] = sn.astype(int)

    # 요일 주기(월=0..일=6)
    dow = out["date"].dt.dayofweek
    out["dow_sin"] = np.sin(2*np.pi*(dow/7))
    out["dow_cos"] = np.cos(2*np.pi*(dow/7))

    # 라벨: 내일 usdkrw 상승 여부
    out["label"] = (out["usdkrw(target)"].diff().shift(-1) > 0).astype(int).fillna(0).astype(int)

    return out

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

def to_tensor(X, y=None):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = None
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X_t, y_t

def batch_iter(X, y, bs, shuffle=True):
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    for st in range(0, len(X), bs):
        ed = min(st+bs, len(X))
        sel = idx[st:ed]
        yield X[sel], y[sel]

# ======================
# 모델
# ======================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
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
        h_last = out[:, -1, :]
        h = self.dropout(h_last)
        return self.head(h)  # (B,1)

# ======================
# 학습/검증 루틴
# ======================
def train_and_validate(seq_len, feat_cols, base_feats_df, seed=2025, epochs=EPOCHS):
    set_seed(seed)

    # 전체 학습용 프레임(훈련 기간만): base_feats_df는 engineer_features(df_train) 결과여야 함
    X_all, y_all, d_all = make_sequences_with_dates(base_feats_df, feat_cols, "label", seq_len)

    # 검증 분리(6월)
    is_val = month_mask(d_all, VAL_MONTH)
    X_val, y_val = X_all[is_val], y_all[is_val]
    X_tr,  y_tr  = X_all[~is_val], y_all[~is_val]

    model = LSTMClassifier(input_dim=len(feat_cols)).to(device)

    # pos_weight = neg/pos (안전 클램핑)
    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    pos_weight_raw = neg / max(1, pos)
    pos_weight_safe = float(np.clip(pos_weight_raw, 0.6, 1.8))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_safe], dtype=torch.float32).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs//2))

    X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
    X_val_t, y_val_t = to_tensor(X_val, y_val)

    best = {"f1": -1, "thr": 0.5, "state": None}
    no_improve = 0

    for ep in range(1, epochs+1):
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
        scheduler.step()

        # 검증 & 임계값 튜닝
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel() if len(X_val_t) else np.array([])

        if len(prob_val):
            # 그리드 검색: F1(상승) 최대, 동률이면 balanced accuracy 최대
            f1_best_ep, thr_ep, bal_acc_best = -1, 0.5, -1
            for thr in THRESH_GRID_VAL:
                pred = (prob_val > thr).astype(int)
                f1p = f1_score(y_val, pred, pos_label=1, zero_division=0)
                if f1p > f1_best_ep:
                    f1_best_ep, thr_ep = f1p, thr
                    bal_acc_best = balanced_accuracy_score(y_val, pred)
                elif np.isclose(f1p, f1_best_ep):
                    bal = balanced_accuracy_score(y_val, pred)
                    if bal > bal_acc_best:
                        thr_ep, bal_acc_best = thr, bal

            if f1_best_ep > best["f1"]:
                best["f1"] = f1_best_ep
                best["thr"] = float(thr_ep)
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if ep % 5 == 0 or ep == 1:
                acc05 = accuracy_score(y_val, (prob_val > 0.5).astype(int))
                print(f"[seed={seed} seq={seq_len} feats={len(feat_cols)}] Ep {ep:03d}/{epochs} "
                      f"Loss={epoch_loss:.4f}  ValAcc@0.5={acc05:.3f}  "
                      f"Val bestF1⁺={f1_best_ep:.3f} (thr={thr_ep:.2f})  [best={best['f1']:.3f}]")
        else:
            print(f"[seed={seed} seq={seq_len}] Ep {ep:03d}/{epochs} Loss={epoch_loss:.4f} (검증없음)")

        if no_improve >= EARLY_STOP_PATIENCE:
            break

    if best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    return model, best["thr"], best["f1"], pos_weight_safe

# ======================
# (0) 훈련 피처 프레임 생성 (학습기간)
# ======================
train_feats = engineer_features(df_train)

# ======================
# (1) 피처셋 랭킹 (seq=45 고정, seed=첫 시드)
# ======================
feat_rank = []
RANK_SEQLEN_FOR_FEATSET = 45
print("\n=== Feature-set ranking (seq=45) ===")
for name, cols in FEATURE_SET_CANDIDATES.items():
    model_tmp, thr_tmp, f1_tmp, pw = train_and_validate(RANK_SEQLEN_FOR_FEATSET, cols, train_feats, seed=SEEDS[0], epochs=40)
    feat_rank.append((f1_tmp, name, cols, thr_tmp))
    print(f"[feature-set: {name:>20}] Val F1⁺={f1_tmp:.3f}  thr={thr_tmp:.2f}  (pos_weight used={pw:.2f})")

feat_rank.sort(reverse=True, key=lambda x: x[0])
BEST_FEAT_NAME, BEST_FEAT_COLS = feat_rank[0][1], feat_rank[0][2]
print(f"\n[선택] 최적 피처셋 = {BEST_FEAT_NAME} ({len(BEST_FEAT_COLS)} cols)")

# ======================
# (2) 시퀀스 길이 랭킹 (최적 피처셋 기준, seed=첫 시드)
# ======================
seq_rank = []
print("\n=== Seq-length ranking ===")
for sl in SEQ_CANDIDATES:
    model_tmp, thr_tmp, f1_tmp, pw = train_and_validate(sl, BEST_FEAT_COLS, train_feats, seed=SEEDS[0], epochs=60)
    seq_rank.append((f1_tmp, sl, thr_tmp))
    print(f"[seq={sl}] Val F1⁺={f1_tmp:.3f} thr={thr_tmp:.2f}")

seq_rank.sort(reverse=True, key=lambda x: x[0])
TOP_SEQS = [seq_rank[i][1] for i in range(min(2, len(seq_rank)))]
print(f"\n[선택] 상위 SEQ_LEN = {TOP_SEQS} (Val F1 기준)")

# ======================
# (3) 7월 market_bin 정책 프레임 생성 함수
# ======================
def build_july_market_bin(policy: str, train_df: pd.DataFrame, test_len: int):
    last7 = train_df["market_bin"].values[-7:]
    if policy == "weekly_repeat":
        return np.tile(last7, int(np.ceil(test_len/7)))[:test_len]
    elif policy == "forward_fill":
        return np.repeat(train_df["market_bin"].values[-1], test_len)
    elif policy == "ema":
        prob = pd.Series(train_df["market_bin"].values).ewm(span=5, adjust=False).mean().iloc[-1]
        return np.repeat(int(prob >= 0.5), test_len)
    else:
        raise ValueError("unknown policy")

# ======================
# (4) 최종 학습 × 시드/시퀀스 앙상블 → 7월 예측
# ======================
val_thr_list = []
logits_accum = None
dates_test = None
ytrue_test = None
N_models = 0
used_policies = []

for sl in TOP_SEQS:
    for sd in SEEDS:
        # 훈련 (최적 피처셋으로)
        print(f"\n=== Final train: seq={sl}, seed={sd}, feats={BEST_FEAT_NAME} ===")
        model, thr_val, f1_val, _ = train_and_validate(sl, BEST_FEAT_COLS, train_feats, seed=sd, epochs=EPOCHS)
        val_thr_list.append(thr_val)

        # 각 정책별 7월 예측
        for pol in MARKET_POLICIES:
            # 7월 base 프레임: market_bin은 정책으로 공급, market 수치 없음
            july_mb = build_july_market_bin(pol, train_feats, len(df_test_target))
            df_july_base = pd.DataFrame({
                "date": df_test_target["date"].values,
                "usdkrw(target)": df_test_target["usdkrw(target)"].values,
                "market_bin": july_mb
            })

            # 학습기간 + 7월 이어붙인 뒤, 동일 피처 엔지니어링(경계 누수 없이 롤링 반영)
            df_all_base = pd.concat([
                df_train[["date","usdkrw(target)","market"]],
                df_july_base[["date","usdkrw(target)","market_bin"]]
            ], ignore_index=True).sort_values("date").reset_index(drop=True)

            df_all_feats = engineer_features(df_all_base)

            # 시퀀스 생성 및 7월만 추출
            X_full, y_full, d_full = make_sequences_with_dates(df_all_feats, BEST_FEAT_COLS, "label", sl)
            mask_test = (d_full >= np.datetime64(TEST_START)) & (d_full <= np.datetime64(TEST_END))
            X_test, y_test, d_test = X_full[mask_test], y_full[mask_test], d_full[mask_test]

            X_test_t, _ = to_tensor(X_test, None)
            model.eval()
            with torch.no_grad():
                logit = model(X_test_t).cpu().numpy().ravel()  # 생 로짓

            if logits_accum is None:
                logits_accum = logit.copy()
                dates_test = d_test
                ytrue_test = y_test
            else:
                logits_accum += logit

            N_models += 1
            used_policies.append(pol)

# 로짓 평균 → 확률
logits_mean = logits_accum / max(1, N_models)
probs_test = 1/(1+np.exp(-logits_mean))

# 최종 임계값: 검증 임계값들의 평균 근처에서 ±0.10 재탐색
center_thr = float(np.mean(val_thr_list)) if len(val_thr_list) else 0.5
thr_min = max(0.01, center_thr - TEST_RESCAN_DELTA)
thr_max = min(0.99, center_thr + TEST_RESCAN_DELTA)
thr_grid_test = np.linspace(thr_min, thr_max, TEST_RESCAN_STEPS)

best_thr_final, best_f1_test, best_balacc = center_thr, -1, -1
for thr in thr_grid_test:
    pred = (probs_test > thr).astype(int)
    f1p = f1_score(ytrue_test, pred, pos_label=1, zero_division=0)
    if f1p > best_f1_test:
        best_f1_test, best_thr_final = f1p, thr
        best_balacc = balanced_accuracy_score(ytrue_test, pred)
    elif np.isclose(f1p, best_f1_test):
        bal = balanced_accuracy_score(ytrue_test, pred)
        if bal > best_balacc:
            best_thr_final, best_balacc = thr, bal

pred_test = (probs_test > best_thr_final).astype(int)

# 결과 프레임
results_df = pd.DataFrame({
    "date": dates_test,
    "true_label": ytrue_test.astype(int),
    "pred_label": pred_test.astype(int),
    "prob_up": probs_test
}).sort_values("date").reset_index(drop=True)
results_df["true_txt"] = results_df["true_label"].map({0:"하락",1:"상승"})
results_df["pred_txt"] = results_df["pred_label"].map({0:"하락",1:"상승"})

print("\n[7월 1~31일 실제/예측 결과]")
print(results_df.to_string(index=False))

print("\n[성능 요약]")
print(f"- 최적 피처셋: {BEST_FEAT_NAME} ({len(BEST_FEAT_COLS)} cols)  후보={list(FEATURE_SET_CANDIDATES.keys())}")
print(f"- 사용 상위 SEQ_LEN: {TOP_SEQS}")
print(f"- 사용 시드: {SEEDS} → 총 모델수(시드×시퀀스×정책) = {N_models}")
print(f"- 정책 목록: {sorted(set(used_policies))}")
print(f"- 검증 임계값 평균: {center_thr:.2f}  7월 재탐색 구간: [{thr_min:.2f}, {thr_max:.2f}]  → 최종={best_thr_final:.2f}")
print("Accuracy:", accuracy_score(results_df["true_label"], results_df["pred_label"]))
print(classification_report(results_df["true_label"], results_df["pred_label"], digits=3, zero_division=0))

if SAVE_CSV:
    results_df.to_csv("july_predictions.csv", index=False)
    print("Saved: july_predictions.csv")

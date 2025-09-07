#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW daily forecasting — TimeXer(LR target) with YF daily exogenous
Data:
  - Train/source: wide_20200101_20250825.csv (≤ 2025-08-25 까지의 일별 데이터)
  - Eval/actuals: df_target_0826.csv (2025-08-26 ~ 2025-09-02 실제값)
Horizon:
  - Predict dates: 2025-08-26 ~ 2025-09-02 (8일)

Notes
- 월/분기/연 저빈도 칼럼은 제외
- 모두 0/상수 칼럼은 제외
- YF 일별(usdkrw/usdjpy/usdcny OHLCV·range·spread·lr 등) 자동 탐지·활용
- 외생 미래 경로: weekday_cycle (최근 7일 패턴 반복)
- tail one-step 보정(bias+amp_scale), EMA10 기반 drift 혼합
"""

import warnings, numpy as np, pandas as pd
from typing import List, Tuple
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

PRED_START = pd.Timestamp('2025-08-26')
PRED_END   = pd.Timestamp('2025-09-02')  # inclusive
SEQ_LEN    = 90
PRED_LEN   = (PRED_END - PRED_START).days + 1  # 8일


# ========================== utils ==========================
def validate_dataframe(df: pd.DataFrame, name: str, cols: list):
    bad = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        n_nan = int(s.isna().sum()); n_inf = int(np.isinf(s).sum())
        if n_nan or n_inf:
            bad[c] = (n_nan, n_inf)
    if bad:
        print(f"[검증 경고] {name}에 비정상 값:")
        for c,(nn,ni) in bad.items():
            print(f"  - {c}: NaN={nn}, Inf={ni}")
    else:
        print(f"[검증 OK] {name}: NaN/Inf 없음")

def clean_infinite_and_nan(df: pd.DataFrame, cols: list, mode="zero"):
    cols = [c for c in cols if c in df.columns]
    df[cols] = df[cols].replace([np.inf,-np.inf], np.nan)
    if mode=="ffill_bfill_then_zero":
        df[cols] = df[cols].ffill().bfill().fillna(0.0)
    else:
        df[cols] = df[cols].fillna(0.0)
    return df

def check_array_finite(name: str, arr: np.ndarray):
    n = arr.size; n_nan=int(np.isnan(arr).sum()); n_inf=int(np.isinf(arr).sum())
    print(f"[배열검증] {name}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf}, valid={(n-n_nan-n_inf)}/{n}")

def drop_constant_or_allzero(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    kept, dropped = [], []
    for c in candidates:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() == 0:
            dropped.append(c); continue
        s = s.fillna(0.0)
        if float(s.std()) == 0.0 or float(np.abs(s).sum()) == 0.0:
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        print(f"[특징 제거] 상수/전부0 칼럼 제거: {len(dropped)}개 → {dropped[:12]}{'...' if len(dropped)>12 else ''}")
    return df.drop(columns=[c for c in dropped if c in df.columns]), kept


# ========================== model ==========================
class SimpleTimeXerModel(nn.Module):
    def __init__(self, endogenous_dim=1, exogenous_dim=5, hidden_size=192, num_layers=2,
                 ar_lags=15, ar_gain=0.90):
        super().__init__()
        self.ar_lags = ar_lags
        self.ar_gain = ar_gain

        self.endogenous_lstm = nn.LSTM(endogenous_dim, hidden_size, num_layers=num_layers,
                                       batch_first=True, dropout=0.1)
        self.exogenous_lstm = nn.LSTM(exogenous_dim, hidden_size, num_layers=num_layers,
                                      batch_first=True, dropout=0.1)

        self.global_token = nn.Parameter(torch.randn(1, hidden_size))
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8,
                                                     dropout=0.1, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)   # neural 1-step r_std (lr)
        )
        self.ar_head = nn.Linear(ar_lags, 1, bias=True)  # linear AR head on r_std window

    def forward(self, endogenous_x, exogenous_x):  # [B,T,1], [B,T,E]
        B, T, _ = endogenous_x.size()

        end_out,_ = self.endogenous_lstm(endogenous_x)   # [B,T,H]
        exo_out,_ = self.exogenous_lstm(exogenous_x)     # [B,T,H]

        query = end_out[:, -1:, :]                       # [B,1,H]
        attn_out,_ = self.cross_attention(query=query, key=exo_out, value=exo_out)  # [B,1,H]
        attn_out = attn_out.squeeze(1)                   # [B,H]

        global_rep = self.global_token.repeat(B,1)       # [B,H]
        neural = self.output_layer(torch.cat([attn_out, global_rep], dim=1))  # [B,1]

        l = min(self.ar_lags, T)
        ar_in = endogenous_x[:, -l:, 0]
        if l < self.ar_lags:
            pad = torch.zeros((B, self.ar_lags - l), device=endogenous_x.device, dtype=endogenous_x.dtype)
            ar_in = torch.cat([pad, ar_in], dim=1)
        ar = self.ar_head(ar_in)                         # [B,1]

        return neural + self.ar_gain * ar


# ========================== technicals ==========================
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# ========================== YF daily detector ==========================
def _detect_yf_daily_columns(cols: List[str]) -> List[str]:
    """
    Yahoo Finance 일별 칼럼 자동 탐지:
    - 접두/접미: usdkrw_*, usdjpy_*, usdcny_*
    - 파생: *_range, *_spread, *_lr 등
    - 타깃은 제외
    """
    prefixes = ('usdkrw_', 'usdjpy_', 'usdcny_')
    daily = []
    for c in cols:
        lc = c.lower()
        if lc.startswith(prefixes) and lc != 'usdkrw(target)':
            daily.append(c)
    whitelist = {'usdkrw_range','usdkrw_spread','usdkrw_lr',
                 'usdjpy_range','usdjpy_spread','usdjpy_lr',
                 'usdcny_range','usdcny_lr'}
    for w in whitelist:
        if w in cols and w not in daily:
            daily.append(w)
    return sorted(list(dict.fromkeys(daily)))


# ========================== data load ==========================
def load_split_wide():
    """
    wide_20200101_20250825.csv에서 2025-08-25까지를 학습 원본으로 사용.
    df_target_0826.csv에서 2025-08-26~09-02 실제값을 읽어 평가에 사용.
    """
    src = pd.read_csv('./RAG/fx_data/wide_20200101_20250825.csv')
    if 'date' not in src.columns:
        raise ValueError("./RAG/fx_data/wide_20200101_20250825.csv에 'date' 칼럼이 필요합니다.")
    src['date'] = pd.to_datetime(src['date'])
    src = src.sort_values('date').reset_index(drop=True)

    # 예측 구간 실제값
    tgt = pd.read_csv('./RAG/prediction_model/df_target_0826.csv')
    if 'date' not in tgt.columns:
        raise ValueError("./RAG/prediction_model/df_target_0826.csv에 'date' 칼럼이 필요합니다.")
    tgt['date'] = pd.to_datetime(tgt['date'])
    tgt = tgt.sort_values('date').reset_index(drop=True)

    # 학습: ≤ 2025-08-25
    train_df = src[src['date'] <= pd.Timestamp('2025-08-25')].copy()
    # 평가 타깃: 2025-08-26 ~ 2025-09-02
    target_df = tgt[(tgt['date'] >= PRED_START) & (tgt['date'] <= PRED_END)].copy()

    # 일부 컬럼명 표준화(있을 때만)
    if 'usdkrw' in train_df.columns and 'usdkrw(target)' not in train_df.columns:
        train_df = train_df.rename(columns={'usdkrw': 'usdkrw(target)'})
    if 'usdkrw' in target_df.columns and 'usdkrw(target)' not in target_df.columns:
        target_df = target_df.rename(columns={'usdkrw': 'usdkrw(target)'})

    if 'usdkrw(target)' not in train_df.columns:
        raise ValueError("학습 데이터에 'usdkrw(target)' 칼럼이 필요합니다.")

    # 타깃 파일에도 실제값이 있어야 성능 계산 가능
    if 'usdkrw(target)' not in target_df.columns:
        raise ValueError("df_target_0826.csv에 'usdkrw(target)' 칼럼이 필요합니다.")

    # 예측 Horizon 길이 체크
    expected_days = (PRED_END - PRED_START).days + 1
    if len(target_df) != expected_days:
        print(f"[경고] df_target_0826.csv의 예측 구간 행수가 {len(target_df)}로, {expected_days}와 다릅니다. 내부적으로 날짜 기준으로 정렬만 수행합니다.")

    return train_df.reset_index(drop=True), target_df.reset_index(drop=True)


# ========================== preprocessing (DAILY only) ==========================
def preprocess_daily(train_df: pd.DataFrame):
    # 저빈도(월/분기/연)로 알려진 칼럼들 제거
    drop_if_exists = [
        'base','us_current','us_growth','us_interest',
        'us_ex','us_im','reserve','us_reserve','us_export','us_import',
        'us_gdp','consumer','exp_rate','im_rate','us_stock','dir_inv','us_indpro','us_unemp','us_prod'
    ]
    to_drop = [c for c in drop_if_exists if c in train_df.columns]
    if to_drop:
        train_df = train_df.drop(columns=to_drop)

    target_col = 'usdkrw(target)'
    y = train_df[target_col].astype(float)

    # lr/EMA10
    lr = np.log(y / y.shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    train_df['lr'] = lr
    train_df['lr_ema10'] = lr.ewm(span=10, adjust=False).mean()

    # 기술지표
    ma20 = y.rolling(20).mean(); std20 = y.rolling(20).std()
    train_df['lvl_z20']   = (y - ma20) / (std20 + 1e-12)
    train_df['lvl_rsi14'] = rsi(y, 14) / 100.0
    train_df['lr_ma5']    = lr.rolling(5).mean()
    train_df['lr_ma10']   = lr.rolling(10).mean()
    train_df['lr_vol10']  = lr.rolling(10).std()
    train_df['lr_ema3']   = lr.ewm(span=3,  adjust=False).mean()
    train_df['lr_vol20']  = lr.rolling(20).std()
    train_df['lvl_mom5']  = (y / y.shift(5) - 1.0)
    train_df['lvl_z60']   = (y - y.rolling(60).mean()) / (y.rolling(60).std() + 1e-12)
    tech_cols = ['lvl_z20','lvl_rsi14','lr_ma5','lr_ma10','lr_vol10','lr_ema3','lr_vol20','lvl_mom5','lvl_z60']

    # 캘린더
    train_df['dow']     = train_df['date'].dt.weekday
    train_df['eom']     = (train_df['date'].dt.is_month_end).astype(int)
    train_df['dow_sin'] = np.sin(2*np.pi*train_df['dow']/7)
    train_df['dow_cos'] = np.cos(2*np.pi*train_df['dow']/7)

    # YF 일별 자동 탐지
    yf_daily_cols = _detect_yf_daily_columns(train_df.columns.tolist())

    # 기존 일별 동적 후보
    base_dynamic = [c for c in ['market'] if c in train_df.columns]

    dynamic_candidates = base_dynamic + yf_daily_cols
    train_df, dynamic_columns = drop_constant_or_allzero(train_df, dynamic_candidates)

    base_cols = [target_col,'lr','lr_ema10','dow_sin','dow_cos','eom'] + dynamic_columns + tech_cols
    validate_dataframe(train_df, "파생 후(원본, 일별+YF)", base_cols)
    train_df = clean_infinite_and_nan(train_df, base_cols, mode="ffill_bfill_then_zero")

    # 스케일러
    return_scaler    = StandardScaler()
    exogenous_scaler = StandardScaler()

    r_std = return_scaler.fit_transform(train_df[['lr']]).ravel()
    df_scaled = pd.DataFrame({'r_std': r_std}, index=train_df.index)

    exog_cols_core = dynamic_columns + ['lr_ema10','dow_sin','dow_cos','eom'] + tech_cols
    exog_scaled = pd.DataFrame(index=train_df.index)
    if exog_cols_core:
        exog_scaled[exog_cols_core] = exogenous_scaler.fit_transform(train_df[exog_cols_core])

    # r_std 래그(1~15)
    for k in range(1, 16):
        df_scaled[f'rstd_lag{k}'] = df_scaled['r_std'].shift(k)

    df_scaled = df_scaled.join(exog_scaled).fillna(0.0)
    exog_cols = [c for c in df_scaled.columns if c != 'r_std']

    # lr 동적 클램프 경계
    tail = train_df['lr'].iloc[-120:] if len(train_df) >= 120 else train_df['lr']
    lr_q_low, lr_q_high = tail.quantile([0.005, 0.995]).tolist()

    start_level = float(train_df[target_col].iloc[-1])
    df_drift = train_df[['lr_ema10']].copy()

    # 로그
    print(f"학습 기간: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"동적 외생(일별; YF 포함): {dynamic_columns[:20]}{'...' if len(dynamic_columns)>20 else ''}")
    print(f"기술지표(일별): {len(tech_cols)}개")
    print(f"exog 채널 수(래그 포함): {len(exog_cols)}")

    return df_scaled, return_scaler, exogenous_scaler, exog_cols, lr_q_low, lr_q_high, start_level, df_drift


# ========================== seq maker ==========================
def create_sequences_multistep_policy(data: pd.DataFrame, seq_length=SEQ_LEN, pred_length=PRED_LEN):
    arr = data.values
    D = arr.shape[1]; E = D - 1
    X_seq, y_seq, X_exo_fut = [], [], []
    for i in range(seq_length, len(data) - pred_length + 1):
        past = arr[i-seq_length:i, :]              # [seq, D]
        past_exo = past[:, 1:] if E>0 else np.zeros((seq_length,0))
        last7 = past_exo[-7:] if len(past_exo) else np.zeros((7,0))
        reps = int(np.ceil(pred_length/7))
        fut_exo = np.vstack([last7]*reps)[:pred_length]
        fut_r = arr[i:i+pred_length, 0]            # r_std truth
        X_seq.append(past); y_seq.append(fut_r); X_exo_fut.append(fut_exo)
    X = np.asarray(X_seq); y = np.asarray(y_seq); X_exo_future=np.asarray(X_exo_fut)
    check_array_finite("학습입력(X_seq)", X)
    check_array_finite("학습타깃(y_std)", y)
    check_array_finite("학습미래외생(X_exo_future)", X_exo_future)
    return X, y, X_exo_future


# ========================== future exog for inference ==========================
def make_future_exog_scaled_from_train(df_train_scaled: pd.DataFrame, exog_cols: List[str],
                                       horizon: int, mode="weekday_cycle") -> np.ndarray:
    if not exog_cols:
        return np.zeros((horizon, 0), dtype=float)
    hist = df_train_scaled[exog_cols].copy().tail(28)  # 최근 4주
    if mode == "weekday_cycle":
        last_7 = hist.tail(7).to_numpy()
        reps = int(np.ceil(horizon/7))
        fut = np.vstack([last_7]*reps)[:horizon]
    elif mode == "weekly_repeat":
        last_7 = hist.tail(7).to_numpy()
        reps = int(np.ceil(horizon/7))
        fut = np.vstack([last_7]*reps)[:horizon]
    elif mode == "biweekly_repeat":
        last_14 = hist.tail(14).to_numpy()
        reps = int(np.ceil(horizon/14))
        fut = np.vstack([last_14]*reps)[:horizon]
    elif mode == "hold_last":
        fut = np.tile(hist.iloc[-1].to_numpy(), (horizon, 1))
    else:
        raise ValueError("Unknown exog mode")
    return np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)


# ========================== training ==========================
def train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                epochs=260, lr=3e-4,
                                tf_start=0.8, tf_end=0.05,
                                gamma_drift=0.002, lam_sign=0.20, lam_trend=0.35, lam_slope=0.15,
                                patience=50):
    print("모델 학습 시작 (policy-aligned exog + TF + AR head)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    model = model.to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    N, seq, D = X_seq.shape
    pred_len = y_std.shape[1]
    E = D - 1

    Xt = torch.tensor(X_seq, dtype=torch.float32, device=device)
    y_all = torch.tensor(y_std, dtype=torch.float32, device=device)
    exo_fut = torch.tensor(X_exo_future, dtype=torch.float32, device=device)

    crit = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best=1e9; bad=0

    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad()
        p_tf = tf_start + (tf_end - tf_start) * ((epoch-1)/(epochs-1))

        cur = Xt.clone()
        preds = []

        for t in range(pred_len):
            endo = cur[:, :, 0:1]
            exog = cur[:, :, 1:] if E>0 else cur[:, :, 0:1]
            out = model(endo, exog)                 # [N,1] r_std one-step
            preds.append(out)

            use_truth = (torch.rand(N, device=device) < p_tf).float().unsqueeze(1)
            next_r = use_truth * y_all[:, t:t+1] + (1.0 - use_truth) * out

            exo_t = exo_fut[:, t, :] if E>0 else torch.zeros((N,0), device=device)
            new_row = torch.cat([next_r, exo_t], dim=1).unsqueeze(1)
            cur = torch.cat([cur, new_row], dim=1)[:, -seq:, :]

        preds = torch.cat(preds, dim=1)  # [N, pred_len]

        # 부가 손실
        cum_pred = torch.cumsum(preds, dim=1)
        cum_true = torch.cumsum(y_all, dim=1)
        trend_loss = nn.MSELoss()(cum_pred, cum_true)
        slope_pred = (preds[:, -1] - preds[:, 0]) / pred_len
        slope_true = (y_all[:, -1] - y_all[:, 0]) / pred_len
        slope_loss = nn.MSELoss()(slope_pred, slope_true)
        sign_loss = 1.0 - torch.mean(torch.sign(preds).eq(torch.sign(y_all)).float())
        drift_penalty = (preds.mean(dim=1, keepdim=True) ** 2).mean()

        loss = (crit(preds, y_all)
                + lam_sign * sign_loss
                + lam_trend * trend_loss
                + lam_slope * slope_loss
                + gamma_drift * drift_penalty)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                dir_acc = torch.mean(torch.sign(preds).eq(torch.sign(y_all)).float()).item()
            print(f"Epoch {epoch:>3}/{epochs}  Loss={loss.item():.6f}  TF={p_tf:.2f}  DirAcc={dir_acc:.3f}")

        if loss.item()<best-1e-6:
            best=loss.item(); bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {epoch} (best loss {best:.6f})")
                break
    print("모델 학습 완료.")
    return model


# ========================== bias & scale + drift ==========================
def estimate_bias_and_scale(df_train_scaled, return_scaler, model, seq_len=SEQ_LEN, lookback=60):
    device = next(model.parameters()).device
    arr = df_train_scaled.to_numpy()
    if len(arr) < seq_len + lookback + 1:
        print("[BiasCalib] 데이터가 부족하여 보정 생략")
        return 0.0, 1.0

    seq = arr[-(seq_len+lookback):-lookback]   # [seq_len, D]
    lr_true_tail = return_scaler.inverse_transform(
        df_train_scaled[['r_std']].iloc[-lookback:].to_numpy()
    ).ravel()

    lr_pred_tail = []
    with torch.no_grad():
        for t in range(lookback):
            xt = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            endo = xt[:, :, 0:1]
            exog = xt[:, :, 1:] if seq.shape[1] > 1 else xt[:, :, 0:1]
            r_std = model(endo, exog).cpu().numpy().ravel()[0]
            lr_hat = return_scaler.inverse_transform([[r_std]]).ravel()[0]
            lr_pred_tail.append(lr_hat)

            # 다음 step 준비(예측 r_std 주입 + 실제 exog)
            exo_t = arr[-lookback + t, 1:] if arr.shape[1] > 1 else np.array([])
            r_std_next = return_scaler.transform([[lr_hat]]).ravel()[0]
            new_row = np.concatenate([[r_std_next], exo_t], axis=0)
            seq = np.vstack([seq[1:], new_row])

    lr_pred_tail = np.asarray(lr_pred_tail)
    bias = float(np.mean(lr_true_tail - lr_pred_tail))
    s_true = float(np.std(lr_true_tail) + 1e-12)
    s_pred = float(np.std(lr_pred_tail) + 1e-12)
    amp_scale = float(np.clip(s_true / s_pred, 0.8, 2.5))
    print(f"[BiasCalib] bias={bias:.3e}, std_true={s_true:.3e}, std_pred={s_pred:.3e}, scale={amp_scale:.3f}")
    return bias, amp_scale

def make_drift_path_from_ema10(df_train_drift: pd.DataFrame, horizon: int) -> np.ndarray:
    last = float(df_train_drift['lr_ema10'].iloc[-1]) if 'lr_ema10' in df_train_drift.columns else 0.0
    return np.full(horizon, last, dtype=float)


# ========================== inference ==========================
def predict_future_daily(model,
                         last_seq_scaled: np.ndarray,
                         return_scaler: StandardScaler,
                         future_exog_scaled: np.ndarray,
                         start_level: float,
                         lr_low: float, lr_high: float,
                         bias: float = 0.0,
                         amp_scale: float = 1.0,
                         drift_path: np.ndarray = None,
                         drift_weight: float = 0.25,
                         shrink_tau: float = None) -> List[float]:
    device = next(model.parameters()).device
    model.eval(); H=len(future_exog_scaled)
    seq = last_seq_scaled.copy(); win = seq.shape[0]
    preds_level=[]; cur=float(start_level)
    if drift_path is None:
        drift_path = np.zeros(H, dtype=float)

    with torch.no_grad():
        for t in range(H):
            xt = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            endo = xt[:, :, 0:1]
            exog = xt[:, :, 1:] if seq.shape[1]>1 else xt[:, :, 0:1]
            r_std = model(endo, exog).cpu().numpy().ravel()[0]
            lr_hat = return_scaler.inverse_transform([[r_std]]).ravel()[0]

            # 보정 + 클램프 + drift 혼합
            lr_hat = (lr_hat + bias) * amp_scale
            lr_hat = float(np.clip(lr_hat, lr_low, lr_high))
            lr_hat = (1.0 - drift_weight) * lr_hat + drift_weight * float(drift_path[t])

            if shrink_tau is not None and shrink_tau > 0:
                lr_hat *= float(np.exp(-t / float(shrink_tau)))

            cur = cur * np.exp(lr_hat)
            preds_level.append(cur)

            exog_t = future_exog_scaled[t] if future_exog_scaled.size else np.array([])
            r_std_next = return_scaler.transform([[lr_hat]]).ravel()[0]
            new_row = np.concatenate([[r_std_next], exog_t], axis=0)
            seq = np.vstack([seq, new_row])[-win:]
    return preds_level


# ========================== plot / report ==========================
def plot_results(actual, predicted, dates, title_suffix=""):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, predicted, '--', label='Predicted', linewidth=2)
    plt.title(f'USD/KRW Prediction {title_suffix}'.strip())
    plt.xlabel('Date'); plt.ylabel('USD/KRW')
    plt.legend(); plt.grid(True, alpha=0.3); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig('timexer_lr_results_0826_0902.png', dpi=300, bbox_inches='tight'); plt.show()

def save_results(actual, predicted, dates, model_name='TimeXer LR + YF (0826~0902)'):
    actual=np.asarray(actual, dtype=float); predicted=np.asarray(predicted, dtype=float)
    predicted = np.nan_to_num(predicted, nan=np.nanmean(predicted) if np.any(~np.isnan(predicted)) else 0.0)
    df=pd.DataFrame({'date':dates,'actual':actual,'predicted':predicted})
    df['error']=df['actual']-df['predicted']; df['error_pct']=df['error']/df['actual']*100.0
    df.to_csv('timexer_lr_results_0826_0902.csv', index=False)
    mse=mean_squared_error(actual,predicted); mae=mean_absolute_error(actual,predicted); rmse=float(np.sqrt(mse))
    dir_acc = np.mean(np.sign(actual[1:]-actual[:-1]) == np.sign(predicted[1:]-predicted[:-1]))
    print("\n예측 성능:"); print(f"MSE : {mse:.4f}"); print(f"MAE : {mae:.4f}"); print(f"RMSE: {rmse:.4f}"); print(f"Direction Acc: {dir_acc:.3f}")
    pd.DataFrame([{'MSE':mse,'MAE':mae,'RMSE':rmse,'DirAcc':dir_acc,
                   'prediction_days':len(predicted),'model':model_name}]
                 ).to_csv('timexer_lr_summary_0826_0902.csv', index=False)
    print("\n저장 완료:\n- timexer_lr_results_0826_0902.csv\n- timexer_lr_summary_0826_0902.csv\n- timexer_lr_results_0826_0902.png")

def baseline_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    naive = np.r_[actual[0], actual[:-1]]
    lr = np.r_[0, np.log(actual[1:]/actual[:-1])]
    ema10 = pd.Series(lr).ewm(span=10, adjust=False).mean().values
    drift = actual[0] * np.exp(np.cumsum(ema10))
    def pr(n):
        return np.sqrt(mean_squared_error(actual, n)), mean_absolute_error(actual, n)
    rmse_m, mae_m = pr(predicted)
    rmse_n, mae_n = pr(naive)
    rmse_d, mae_d = pr(drift)
    dir_acc = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted)))
    corr = np.corrcoef(actual, predicted)[0,1]
    print(f"[Model ] RMSE={rmse_m:.3f} MAE={mae_m:.3f}  DirAcc={dir_acc:.3f}  Corr={corr:.3f}")
    print(f"[Naive1] RMSE={rmse_n:.3f} MAE={mae_n:.3f}")
    print(f"[Drift ] RMSE={rmse_d:.3f} MAE={mae_d:.3f}")


# ========================== main ==========================
def main():
    print("="*74)
    print("Predict 2025-08-26 ~ 2025-09-02 using wide_20200101_20250825.csv + df_target_0826.csv")
    print("="*74)

    # 1) 데이터 로드/분할
    train_raw, target_raw = load_split_wide()
    target_dates = target_raw['date'].values
    actual_values = target_raw['usdkrw(target)'].to_numpy()

    # 2) 전처리/스케일
    (df_train_scaled, return_scaler, exogenous_scaler,
     exog_cols, lr_q_low, lr_q_high, start_level, df_train_drift) = preprocess_daily(train_raw)

    # 3) 학습 세트 구성 및 학습
    X_seq, y_std, X_exo_future = create_sequences_multistep_policy(
        df_train_scaled, seq_length=SEQ_LEN, pred_length=PRED_LEN
    )
    print(f"학습 시퀀스: {X_seq.shape}, 타깃(r_std): {y_std.shape}, 미래외생: {X_exo_future.shape}")

    D = df_train_scaled.shape[1]; exog_dim = max(1, D-1)
    model = SimpleTimeXerModel(endogenous_dim=1, exogenous_dim=exog_dim,
                               hidden_size=192, num_layers=2, ar_lags=15, ar_gain=0.90)
    model = train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                        epochs=260, lr=3e-4, tf_start=0.8, tf_end=0.05,
                                        gamma_drift=0.002, lam_sign=0.20, lam_trend=0.35, lam_slope=0.15,
                                        patience=50)

    # 4) 예측 준비
    last_seq_scaled = df_train_scaled.iloc[-SEQ_LEN:].to_numpy()
    check_array_finite("마지막시퀀스(last_seq_scaled)", last_seq_scaled)

    H = PRED_LEN
    future_exog_scaled = make_future_exog_scaled_from_train(
        df_train_scaled, [c for c in df_train_scaled.columns if c!='r_std'], H, mode="weekday_cycle"
    )
    check_array_finite("타깃외생(future_exog_scaled)", future_exog_scaled)

    # 5) tail 기반 bias/scale + drift
    bias, amp_scale = estimate_bias_and_scale(df_train_scaled, return_scaler, model,
                                              seq_len=SEQ_LEN, lookback=60)
    drift_path = make_drift_path_from_ema10(df_train_drift, H)

    # 6) 예측
    preds = predict_future_daily(model, last_seq_scaled, return_scaler,
                                 future_exog_scaled, start_level=start_level,
                                 lr_low=lr_q_low, lr_high=lr_q_high,
                                 bias=bias, amp_scale=amp_scale,
                                 drift_path=drift_path, drift_weight=0.25,
                                 shrink_tau=None)

    # 7) 결과/리포트
    print("\n예측 결과 요약:")
    print(f"실제값 범위: {actual_values.min():.1f} ~ {actual_values.max():.1f}")
    print(f"예측값 범위: {np.nanmin(preds):.1f} ~ {np.nanmax(preds):.1f}")

    plot_results(actual_values, preds, target_dates,
                 title_suffix="(2025-08-26 ~ 2025-09-02)")
    save_results(actual_values, preds, target_dates)
    baseline_metrics(actual_values, preds)

    print("\n완료")


if __name__ == "__main__":
    main()

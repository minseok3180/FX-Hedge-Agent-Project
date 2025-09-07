#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 주입용 USD/KRW 주간(7일) 예측 스크립트
- Train: ≤ 마지막 관측일
- Predict: 다음 7일 (T+1 ~ T+7)
- 평가/플롯 없음
- 일별 특징 + YF 일별 칼럼 자동 탐지 + 선택적 야후 외생 병합
- 상수/전부0 칼럼 자동 제거
- 미래 외생: weekday_cycle(최근 7일 반복)
- tail 기반 bias/amp 보정 + EMA10 drift 혼합

Usage:
  # 기준일(as-of) = 2025-09-04 → 파일명: predicted_20250904.csv / .json
  # 저장 위치: 기본값은 이 파이썬 파일이 있는 폴더
  python rag_weekly_predict.py \
    --fx wide_20200101_20250904.csv \
    --target_col "usdkrw(target)" \
    --use_yahoo 1 \
    --horizon 7 \
    --seq_len 90 \
    --asof_date 2025-09-04
"""

import argparse, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ========================== 작은 유틸 ==========================
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
        print(f"[검증 경고] {name}에 NaN/Inf 존재:")
        for c,(nn,ni) in bad.items():
            print(f"  - {c}: NaN={nn}, Inf={ni}")
    else:
        print(f"[검증 OK] {name}: NaN/Inf 없음")

def clean_infinite_and_nan(df: pd.DataFrame, cols: list, mode="ffill_bfill_then_zero"):
    cols = [c for c in cols if c in df.columns]
    df[cols] = df[cols].replace([np.inf,-np.inf], np.nan)
    if mode=="ffill_bfill_then_zero":
        df[cols] = df[cols].ffill().bfill().fillna(0.0)
    else:
        df[cols] = df[cols].fillna(0.0)
    return df

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
        print(f"[특징 제거] 상수/전부0 칼럼 {len(dropped)}개: {dropped[:12]}{'...' if len(dropped)>12 else ''}")
    return df.drop(columns=[c for c in dropped if c in df.columns]), kept

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _detect_yf_daily_columns(cols: List[str]) -> List[str]:
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

# ========================== 선택적 야후 외생 ==========================
def fetch_yahoo_daily(start, end, tickers=None):
    """
    - 날짜 tz-naive 유지
    - 기본 tickers: UUP, ^TNX, ^VIX, ^DXY (개별 실패 무시)
    """
    if tickers is None:
        tickers = ["UUP", "^TNX", "^VIX", "^DXY"]

    try:
        import yfinance as yf
    except Exception:
        print("[야후] yfinance 미설치 또는 임포트 실패 → 외생 건너뜀")
        return pd.DataFrame()

    outs = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end)
            if h.empty:
                continue
            h = h.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            h[f"{t}_close"] = h["close"]
            h[f"{t}_lr"] = np.log(h["close"] / h["close"].shift(1)).replace([np.inf,-np.inf], np.nan)
            out = h[[f"{t}_close", f"{t}_lr"]].copy()
            out.index = out.index.tz_localize(None)
            out = out.reset_index().rename(columns={"Date":"date"})
            outs.append(out)
        except Exception:
            continue

    if not outs:
        return pd.DataFrame()

    df = outs[0]
    for add in outs[1:]:
        df = df.merge(add, on="date", how="outer")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date")

# ========================== 모델 ==========================
class SimpleTimeXerModel(nn.Module):
    def __init__(self, endogenous_dim=1, exogenous_dim=5, hidden_size=160, num_layers=2,
                 ar_lags=15, ar_gain=0.90):
        super().__init__()
        self.ar_lags = ar_lags
        self.ar_gain = ar_gain

        self.endogenous_lstm = nn.LSTM(endogenous_dim, hidden_size, num_layers=num_layers,
                                       batch_first=True, dropout=0.1)
        self.exogenous_lstm = nn.LSTM(max(exogenous_dim,1), hidden_size, num_layers=num_layers,
                                      batch_first=True, dropout=0.1)

        self.cross = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8,
                                           dropout=0.1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        self.ar_head = nn.Linear(ar_lags, 1, bias=True)

    def forward(self, endo_seq, exo_seq):  # [B,T,1], [B,T,E]
        B, T, _ = endo_seq.size()
        e_out,_ = self.endogenous_lstm(endo_seq)       # [B,T,H]
        x_out,_ = self.exogenous_lstm(exo_seq if exo_seq.size(-1)>0 else endo_seq)
        q = e_out[:, -1:, :]                           # [B,1,H]
        attn,_ = self.cross(q, x_out, x_out)           # [B,1,H]
        g = attn.squeeze(1)                            # [B,H]
        z = torch.cat([g, g.detach()], dim=1)          # 간단 연결
        neural = self.head(z)                          # [B,1]

        l = min(self.ar_lags, T)
        ar_in = endo_seq[:, -l:, 0]
        if l < self.ar_lags:
            pad = torch.zeros((B, self.ar_lags - l), device=endo_seq.device, dtype=endo_seq.dtype)
            ar_in = torch.cat([pad, ar_in], dim=1)
        ar = self.ar_head(ar_in)                       # [B,1]
        return neural + self.ar_gain * ar

# ========================== 전처리(일별) ==========================
LOW_FREQ_DROP = [
    'base','us_current','us_growth','us_interest',
    'us_ex','us_im','reserve','us_reserve','us_export','us_import',
    'us_gdp','consumer','exp_rate','im_rate','us_stock','dir_inv','us_indpro','us_unemp','us_prod'
]

def build_daily_dataset(fx_csv: Path, target_col: str, use_yahoo: bool):
    df = pd.read_csv(fx_csv)
    if 'date' not in df.columns:
        raise ValueError("입력 CSV에 'date' 칼럼이 필요합니다.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # 타깃명 표준화
    if target_col not in df.columns and 'usdkrw' in df.columns and target_col == 'usdkrw(target)':
        df = df.rename(columns={'usdkrw': 'usdkrw(target)'})
    if target_col not in df.columns:
        raise ValueError(f"타깃 칼럼 '{target_col}' 이(가) 없습니다.")

    # 저빈도 제거
    to_drop = [c for c in LOW_FREQ_DROP if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # 기본 파생
    y = df[target_col].astype(float)
    lr = np.log(y / y.shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    df['lr'] = lr
    df['lr_ema10'] = lr.ewm(span=10, adjust=False).mean()
    df['lvl_z20'] = (y - y.rolling(20).mean()) / (y.rolling(20).std() + 1e-12)
    df['lvl_rsi14'] = rsi(y, 14) / 100.0
    df['lr_ma5'] = lr.rolling(5).mean()
    df['lr_ma10'] = lr.rolling(10).mean()
    df['lr_vol10'] = lr.rolling(10).std()
    df['lr_ema3'] = lr.ewm(span=3, adjust=False).mean()
    df['lr_vol20'] = lr.rolling(20).std()
    df['lvl_mom5'] = (y / y.shift(5) - 1.0)
    df['lvl_z60'] = (y - y.rolling(60).mean()) / (y.rolling(60).std() + 1e-12)

    # 캘린더
    df['dow'] = df['date'].dt.weekday
    df['eom'] = (df['date'].dt.is_month_end).astype(int)
    df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)

    # 원본 내 YF 일별 자동 탐지
    yf_daily_cols = _detect_yf_daily_columns(df.columns.tolist())

    # 선택적 야후 병합
    exo_add = []
    if use_yahoo:
        ystart = (df['date'].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        yend   = (df['date'].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        ydf = fetch_yahoo_daily(ystart, yend)
        if not ydf.empty:
            ydf['date'] = pd.to_datetime(ydf['date'], errors='coerce')
            ydf = ydf.dropna(subset=['date'])
            df = df.merge(ydf, on='date', how='left')
            exo_add = [c for c in ydf.columns if c != 'date']
        else:
            print("[야후] 병합할 데이터가 없어 건너뜀")

    # 동적 외생 후보(전부 일별)
    base_dynamic = [c for c in ['market'] if c in df.columns]
    dynamic_candidates = base_dynamic + yf_daily_cols + exo_add

    # 상수/전부0 제거
    df, dynamic_cols = drop_constant_or_allzero(df, dynamic_candidates)

    # 스케일러
    return_scaler    = StandardScaler()
    exogenous_scaler = StandardScaler()

    r_std = return_scaler.fit_transform(df[['lr']]).ravel()
    df_scaled = pd.DataFrame({'r_std': r_std}, index=df.index)

    tech_cols = ['lvl_z20','lvl_rsi14','lr_ma5','lr_ma10','lr_vol10','lr_ema3','lr_vol20','lvl_mom5','lvl_z60']
    exo_core = dynamic_cols + ['lr_ema10','dow_sin','dow_cos','eom'] + tech_cols
    exo_core = [c for c in exo_core if c in df.columns]
    if exo_core:
        exo_scaled = exogenous_scaler.fit_transform(df[exo_core].copy())
        exo_scaled = pd.DataFrame(exo_scaled, columns=exo_core, index=df.index)
        df_scaled = df_scaled.join(exo_scaled, how='left')
    else:
        exo_core = []

    # r_std 래그(1~15)
    for k in range(1, 16):
        df_scaled[f'rstd_lag{k}'] = pd.Series(r_std, index=df.index).shift(k)

    df_scaled = df_scaled.fillna(0.0)

    # 클램프 경계(최근 120일)
    tail = df['lr'].iloc[-120:] if len(df) >= 120 else df['lr']
    lr_q_low, lr_q_high = tail.quantile([0.005, 0.995]).tolist()

    start_level = float(df[target_col].iloc[-1])
    drift_df = df[['lr_ema10']].copy()

    print(f"학습 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"동적 외생(일별; YF 포함): {dynamic_cols[:20]}{'...' if len(dynamic_cols)>20 else ''}")
    print(f"기술지표(일별): {len(tech_cols)}개")
    print(f"exog 채널 수(래그 제외): {len(exo_core)}; 래그 포함 총: {df_scaled.shape[1]-1}")

    return df, df_scaled, return_scaler, exogenous_scaler, exo_core, lr_q_low, lr_q_high, start_level, drift_df

# ========================== 학습용 시퀀스 ==========================
def make_seq(data: pd.DataFrame, seq_len: int, pred_len: int):
    arr = data.to_numpy(dtype=np.float32)         # [N, D]
    D = arr.shape[1]; E = D - 1
    X_seq, y_seq, X_exo_fut = [], [], []
    for i in range(seq_len, len(arr) - pred_len + 1):
        past = arr[i-seq_len:i, :]                # [seq, D]
        past_exo = past[:, 1:] if E>0 else np.zeros((seq_len, 0), dtype=np.float32)
        last7 = past_exo[-7:] if len(past_exo) else np.zeros((7,0), dtype=np.float32)
        reps = int(np.ceil(pred_len/7))
        fut_exo = np.vstack([last7]*reps)[:pred_len]
        fut_r = arr[i:i+pred_len, 0]              # true r_std
        X_seq.append(past); y_seq.append(fut_r); X_exo_fut.append(fut_exo)
    return np.asarray(X_seq), np.asarray(y_seq), np.asarray(X_exo_fut)

# ========================== 보정/드리프트 ==========================
def estimate_bias_scale(df_scaled: pd.DataFrame, return_scaler: StandardScaler,
                        model: nn.Module, seq_len=90, lookback=60):
    device = next(model.parameters()).device
    arr = df_scaled.to_numpy(dtype=np.float32)
    if len(arr) < seq_len + lookback + 1:
        print("[BiasCalib] 데이터 부족 → 보정 생략")
        return 0.0, 1.0
    seq = arr[-(seq_len+lookback):-lookback]
    lr_true_tail = return_scaler.inverse_transform(
        df_scaled[['r_std']].iloc[-lookback:].to_numpy()
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
            exo_t = arr[-lookback + t, 1:] if arr.shape[1] > 1 else np.array([], dtype=np.float32)
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

def make_drift_path(drift_df: pd.DataFrame, horizon: int):
    last = float(drift_df['lr_ema10'].iloc[-1]) if 'lr_ema10' in drift_df.columns else 0.0
    return np.full(horizon, last, dtype=float)

# ========================== 학습 루프 ==========================
def train_model(model, X_seq, y_std, X_exo_future,
                epochs=160, lr=3e-4, tf_start=0.8, tf_end=0.08,
                lam_sign=0.20, lam_trend=0.30, lam_slope=0.12, gamma_drift=0.002,
                patience=35):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    model = model.to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    Xt = torch.tensor(X_seq, dtype=torch.float32, device=device)
    y_all = torch.tensor(y_std, dtype=torch.float32, device=device)
    exo_fut = torch.tensor(X_exo_future, dtype=torch.float32, device=device)

    pred_len = y_all.shape[1]
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
            exog = cur[:, :, 1:] if cur.size(-1)>1 else cur[:, :, 0:1]
            out = model(endo, exog)
            preds.append(out)

            use_truth = (torch.rand(Xt.size(0), device=device) < p_tf).float().unsqueeze(1)
            next_r = use_truth * y_all[:, t:t+1] + (1.0 - use_truth) * out

            exo_t = exo_fut[:, t, :]
            new_row = torch.cat([next_r, exo_t], dim=1).unsqueeze(1)
            cur = torch.cat([cur, new_row], dim=1)[:, -Xt.size(1):, :]

        preds = torch.cat(preds, dim=1)

        cum_pred = torch.cumsum(preds, dim=1)
        cum_true = torch.cumsum(y_all, dim=1)
        trend_loss = nn.MSELoss()(cum_pred, cum_true)
        slope_pred = (preds[:, -1] - preds[:, 0]) / pred_len
        slope_true = (y_all[:, -1] - y_all[:, 0]) / pred_len
        slope_loss = nn.MSELoss()(slope_pred, slope_true)
        sign_loss = 1.0 - torch.mean(torch.sign(preds).eq(torch.sign(y_all)).float())
        drift_penalty = (preds.mean(dim=1, keepdim=True) ** 2).mean()

        loss = (crit(preds, y_all)
                + 0.20 * sign_loss
                + 0.30 * trend_loss
                + 0.12 * slope_loss
                + 0.002 * drift_penalty)

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
                print(f"Early stop at epoch {epoch} (best {best:.6f})")
                break
    print("학습 완료.")
    return model

# ========================== 예측 ==========================
def make_future_exog_from_train(df_train_scaled: pd.DataFrame, exog_cols: List[str],
                                horizon: int, mode="weekday_cycle") -> np.ndarray:
    if not exog_cols:
        return np.zeros((horizon, 0), dtype=np.float32)
    hist = df_train_scaled[exog_cols].copy().tail(28)
    if mode == "weekday_cycle":
        last_7 = hist.tail(7).to_numpy()
        reps = int(np.ceil(horizon/7))
        fut = np.vstack([last_7]*reps)[:horizon]
    else:
        last_7 = hist.tail(7).to_numpy()
        reps = int(np.ceil(horizon/7))
        fut = np.vstack([last_7]*reps)[:horizon]
    return np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def predict_horizon(model,
                    last_seq_scaled: np.ndarray,
                    return_scaler: StandardScaler,
                    future_exog_scaled: np.ndarray,
                    start_level: float,
                    lr_low: float, lr_high: float,
                    bias: float = 0.0,
                    amp_scale: float = 1.0,
                    drift_path: np.ndarray = None,
                    drift_weight: float = 0.25,
                    shrink_tau: float = None):
    device = next(model.parameters()).device
    model.eval()
    H = future_exog_scaled.shape[0]
    seq = last_seq_scaled.copy()
    win = seq.shape[0]
    preds_level=[]; preds_lr=[]
    cur=float(start_level)
    if drift_path is None:
        drift_path = np.zeros(H, dtype=float)

    with torch.no_grad():
        for t in range(H):
            xt = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            endo = xt[:, :, 0:1]
            exog = xt[:, :, 1:] if seq.shape[1]>1 else xt[:, :, 0:1]
            r_std = model(endo, exog).cpu().numpy().ravel()[0]
            lr_hat = return_scaler.inverse_transform([[r_std]]).ravel()[0]

            lr_hat = (lr_hat + bias) * amp_scale
            lr_hat = float(np.clip(lr_hat, lr_low, lr_high))
            lr_hat = (1.0 - drift_weight) * lr_hat + drift_weight * float(drift_path[t])
            if shrink_tau is not None and shrink_tau > 0:
                lr_hat *= float(np.exp(-t / float(shrink_tau)))

            cur = cur * np.exp(lr_hat)
            preds_level.append(cur); preds_lr.append(lr_hat)

            exog_t = future_exog_scaled[t] if future_exog_scaled.size else np.array([], dtype=np.float32)
            r_std_next = return_scaler.transform([[lr_hat]]).ravel()[0]
            new_row = np.concatenate([[r_std_next], exog_t], axis=0)
            seq = np.vstack([seq, new_row])[-win:]
    return preds_level, preds_lr

# ========================== 메인 ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", required=True, help="ECOS 와이드 CSV")
    ap.add_argument("--target_col", default="usdkrw(target)")
    ap.add_argument("--use_yahoo", type=int, default=1)
    ap.add_argument("--out_dir", default=None, help="저장 폴더(미입력 시 이 스크립트 파일이 위치한 폴더)")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--seq_len", type=int, default=90)
    ap.add_argument("--asof_date", default=None, help="파일명 기준일 YYYY-MM-DD (미입력 시 데이터 마지막 날짜 사용)")
    args = ap.parse_args()

    # 저장 경로 결정: 기본은 스크립트 디렉터리
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fx_csv = Path(args.fx)

    # 1) 데이터 구성
    raw_df, df_scaled, ret_scaler, exo_scaler, exo_core, lr_low, lr_high, start_level, drift_df = \
        build_daily_dataset(fx_csv, args.target_col, bool(args.use_yahoo))

    # 2) 학습 세트
    X_seq, y_std, X_exo_future = make_seq(df_scaled, seq_len=args.seq_len, pred_len=args.horizon)
    print(f"학습 시퀀스: {X_seq.shape}, 타깃(r_std): {y_std.shape}, 미래외생: {X_exo_future.shape}")

    D = df_scaled.shape[1]; exog_dim = max(1, D-1)
    model = SimpleTimeXerModel(endogenous_dim=1, exogenous_dim=exog_dim,
                               hidden_size=160, num_layers=2, ar_lags=15, ar_gain=0.90)

    model = train_model(model, X_seq, y_std, X_exo_future,
                        epochs=160, lr=3e-4, tf_start=0.8, tf_end=0.08,
                        lam_sign=0.20, lam_trend=0.30, lam_slope=0.12, gamma_drift=0.002,
                        patience=35)

    # 3) 예측 준비
    last_seq_scaled = df_scaled.iloc[-args.seq_len:].to_numpy(dtype=np.float32)
    fut_exog_scaled = make_future_exog_from_train(
        df_scaled, [c for c in df_scaled.columns if c!='r_std'], args.horizon, mode="weekday_cycle"
    )
    bias, amp = estimate_bias_scale(df_scaled, ret_scaler, model,
                                    seq_len=args.seq_len, lookback=min(60, len(df_scaled)-args.seq_len-1))
    drift_path = make_drift_path(drift_df, args.horizon)

    # 4) 예측 실행
    preds_level, preds_lr = predict_horizon(
        model, last_seq_scaled, ret_scaler, fut_exog_scaled,
        start_level=start_level, lr_low=lr_low, lr_high=lr_high,
        bias=bias, amp_scale=amp, drift_path=drift_path, drift_weight=0.25, shrink_tau=None
    )

    # 5) 저장 파일명 결정
    last_day = pd.to_datetime(raw_df['date'].iloc[-1])
    if args.asof_date:
        try:
            asof_dt = pd.to_datetime(args.asof_date)
        except Exception:
            print(f"[경고] --asof_date 파싱 실패: {args.asof_date} → 데이터 마지막 날짜로 대체")
            asof_dt = last_day
    else:
        asof_dt = last_day

    if asof_dt.date() != last_day.date():
        print(f"[경고] --asof_date({asof_dt.date()}) ≠ 데이터 마지막 날짜({last_day.date()}). "
              f"예측 기준은 데이터 마지막 날짜를 사용합니다(파일명만 asof 반영).")

    file_stamp = asof_dt.strftime("%Y%m%d")  # 예: 20250904

    # 예측 범위 날짜
    dates = pd.date_range(last_day + pd.Timedelta(days=1), periods=args.horizon, freq='D')

    csv_path = out_dir / f"predicted_{file_stamp}.csv"
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "predicted": np.asarray(preds_level, dtype=float),
        "predicted_lr": np.asarray(preds_lr, dtype=float),
        "source": "rag_weekly_predict.py"
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = out_dir / f"predicted_{file_stamp}.json"
    meta = {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "horizon_days": int(args.horizon),
        "predicted": [
            {"date": d.strftime("%Y-%m-%d"), "predicted": float(y), "predicted_lr": float(r)}
            for d, y, r in zip(dates, preds_level, preds_lr)
        ],
        "exogenous_used": [c for c in df_scaled.columns if c!='r_std'],
        "exo_policy": "weekday_cycle(last7)",
        "bias_correction": {"bias": float(bias), "amp_scale": float(amp)},
        "drift": {"type": "ema10", "weight": 0.25}
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n예측 저장 완료:\n- {csv_path}\n- {json_path}")

if __name__ == "__main__":
    main()

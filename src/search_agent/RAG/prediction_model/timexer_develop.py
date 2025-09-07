#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW daily forecasting (full script, improved)
- Target: lr_resid = lr - EMA(lr, span=10)  → 표준화 후 학습
- Restore: lr̂ = EMA_pred + lr̂_resid  (EMA는 예측 중 갱신)
- Model: LSTM(endo/exo) + cross-attn(T-length) + MLP + linear AR head (lags=15, gain=0.6)
- Train: multi-step teacher forcing, 외생 '최근 7일 반복' 정책(훈련=예측)
- Loss: Huber + 방향성(sign) + 누적 경로(trend) + 기울기(slope) + 약한 drift penalty
- Features: MoM/YoY(low-freq), momentum/vol(resid), level z-score/RSI, weekday/eom sin/cos
- Inference: resid 분위수 클램프(1~99%) + 완만한 shrinkage(τ=14); y = y * exp(lr̂) 누적
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

# ========================== utils ==========================
def validate_dataframe(df: pd.DataFrame, name: str, cols: list):
    bad = {}
    for c in cols:
        if c not in df.columns: continue
        s = df[c]
        n_nan = int(s.isna().sum()); n_inf = int(np.isinf(s).sum())
        if n_nan or n_inf: bad[c] = (n_nan, n_inf)
    if bad:
        print(f"[검증 경고] {name}에 비정상 값:")
        for c,(nn,ni) in bad.items(): print(f"  - {c}: NaN={nn}, Inf={ni}")
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

def _pct_change_safe(s: pd.Series, periods:int) -> pd.Series:
    prev = s.shift(periods)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (s - prev) / prev
    return pct.replace([np.inf,-np.inf], np.nan)

def check_array_finite(name: str, arr: np.ndarray):
    n = arr.size; n_nan=int(np.isnan(arr).sum()); n_inf=int(np.isinf(arr).sum())
    print(f"[배열검증] {name}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf}, valid={(n-n_nan-n_inf)}/{n}")

# ========================== model ==========================
class SimpleTimeXerModel(nn.Module):
    def __init__(self, endogenous_dim=1, exogenous_dim=5, hidden_size=128, num_layers=2,
                 ar_lags=15, ar_gain=0.60):
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
            nn.Linear(hidden_size, 1)   # neural 1-step r_std (resid)
        )
        self.ar_head = nn.Linear(ar_lags, 1, bias=True)  # linear AR head on r_std window

    def forward(self, endogenous_x, exogenous_x):  # [B,T,1], [B,T,E]
        B, T, _ = endogenous_x.size()

        end_out,_ = self.endogenous_lstm(endogenous_x)   # [B,T,H]
        exo_out,_ = self.exogenous_lstm(exogenous_x)     # [B,T,H]

        # 쿼리는 end_out의 마지막 스텝, 키/밸류는 exo_out의 전체 타임스텝
        query = end_out[:, -1:, :]                       # [B,1,H]
        key   = exo_out                                  # [B,T,H]
        value = exo_out                                  # [B,T,H]
        attn_out,_ = self.cross_attention(query=query, key=key, value=value)  # [B,1,H]
        attn_out = attn_out.squeeze(1)                   # [B,H]

        global_rep = self.global_token.repeat(B,1)       # [B,H]
        neural = self.output_layer(torch.cat([attn_out, global_rep], dim=1))  # [B,1]

        # AR head on last lags of endogenous (already standardized resid in input)
        l = min(self.ar_lags, T)
        ar_in = endogenous_x[:, -l:, 0]                  # [B, l]
        if l < self.ar_lags:
            pad = torch.zeros((B, self.ar_lags - l), device=endogenous_x.device, dtype=endogenous_x.dtype)
            ar_in = torch.cat([pad, ar_in], dim=1)
        ar = self.ar_head(ar_in)                         # [B,1]

        return neural + self.ar_gain * ar

# ========================== low-freq features ==========================
def add_lowfreq_change_features(df_daily: pd.DataFrame, date_col: str,
                                monthly_cols: List[str], annual_cols: List[str],
                                prefix_m="mom_", prefix_y="yoy_") -> Tuple[pd.DataFrame, List[str]]:
    df = df_daily.sort_values(date_col).copy()
    df["year"]=df[date_col].dt.year; df["month"]=df[date_col].dt.month
    gen_cols: List[str] = []

    if monthly_cols:
        mdf = (df[[date_col,"year","month"]+monthly_cols]
               .drop_duplicates(subset=["year","month"])
               .groupby(["year","month"],as_index=False).first().sort_values(["year","month"]))
        mdf["anchor_date"] = pd.to_datetime(dict(year=mdf["year"], month=mdf["month"], day=1))
        blocks=[]
        for col in monthly_cols:
            mom = _pct_change_safe(mdf[col].astype(float),1).fillna(0.0)
            nm=f"{prefix_m}{col}"; blocks.append(pd.DataFrame({"anchor_date":mdf["anchor_date"], nm:mom})); gen_cols.append(nm)
        merged=blocks[0]
        for b in blocks[1:]: merged=merged.merge(b,on="anchor_date",how="outer")
        df = df.merge(merged.rename(columns={"anchor_date":date_col}), on=date_col, how="left")
        df[gen_cols] = df[gen_cols].ffill().bfill().fillna(0.0)

    if annual_cols:
        ydf = (df[[date_col,"year"]+annual_cols]
               .drop_duplicates(subset=["year"])
               .groupby(["year"],as_index=False).first().sort_values(["year"]))
        ydf["anchor_date"] = pd.to_datetime(dict(year=ydf["year"], month=1, day=1))
        blocks=[]; ycols=[]
        for col in annual_cols:
            yoy = _pct_change_safe(ydf[col].astype(float),1).fillna(0.0)
            nm=f"{prefix_y}{col}"; blocks.append(pd.DataFrame({"anchor_date":ydf["anchor_date"], nm:yoy}))
            gen_cols.append(nm); ycols.append(nm)
        merged=blocks[0]
        for b in blocks[1:]: merged=merged.merge(b,on="anchor_date",how="outer")
        df = df.merge(merged.rename(columns={"anchor_date":date_col}), on=date_col, how="left")
        df[ycols] = df[ycols].ffill().bfill().fillna(0.0)

    return df.drop(columns=["year","month"], errors="ignore"), gen_cols

# ========================== technicals ==========================
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

# ========================== data load / preprocess ==========================
def load_and_preprocess_data():
    print("데이터 로드 중...")
    df_train = pd.read_csv('df.csv'); df_train['date']=pd.to_datetime(df_train['date'])
    df_train = df_train.sort_values('date').reset_index(drop=True)

    df_target = pd.read_csv('df_target.csv'); df_target['date']=pd.to_datetime(df_target['date'])
    df_target = df_target.sort_values('date').reset_index(drop=True)

    # drop empty cols if exist
    drop_if_exists=['us_current','us_growth','us_interest']
    drop_real=[c for c in drop_if_exists if c in df_train.columns]
    if drop_real:
        df_train=df_train.drop(columns=drop_real)
        df_target=df_target.drop(columns=[c for c in drop_real if c in df_target.columns], errors='ignore')

    dynamic_columns=[c for c in ['base','market','consumer','exp_rate','im_rate'] if c in df_train.columns]
    monthly_like=[c for c in ['us_ex','us_im','reserve','us_reserve','us_export','us_import','us_gdp','us_stock','dir_inv','us_indpro','us_unemp','us_prod'] if c in df_train.columns]
    annual_like: List[str] = []

    df_train_ext, lowfreq_feats = add_lowfreq_change_features(df_train,'date', monthly_like, annual_like)

    target_col='usdkrw(target)'

    # --- log-return + EMA 잔차 타깃 ---
    y = df_train_ext[target_col].astype(float)
    lr = np.log(y / y.shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)

    ema10 = pd.Series(lr).ewm(span=10, adjust=False).mean().values
    df_train_ext['lr_ema10']   = ema10
    df_train_ext['lr_resid']   = lr - ema10  # 학습 타깃(비정상성 완화)

    # momentum/vol은 resid 기준
    r = df_train_ext['lr_resid']
    df_train_ext['r_ma3']   = r.rolling(3).mean()
    df_train_ext['r_ma5']   = r.rolling(5).mean()
    df_train_ext['r_ma10']  = r.rolling(10).mean()
    df_train_ext['r_vol10'] = r.rolling(10).std()

    # level-based reversal
    ma20 = y.rolling(20).mean(); std20 = y.rolling(20).std()
    df_train_ext['lvl_z20']   = (y - ma20) / (std20 + 1e-12)
    df_train_ext['lvl_rsi14'] = rsi(y, 14) / 100.0  # 0~1

    tech_cols=['r_ma3','r_ma5','r_ma10','r_vol10','lvl_z20','lvl_rsi14']

    # calendar
    df_train_ext['dow']     = df_train_ext['date'].dt.weekday
    df_train_ext['eom']     = (df_train_ext['date'].dt.is_month_end).astype(int)
    df_train_ext['dow_sin'] = np.sin(2*np.pi*df_train_ext['dow']/7)
    df_train_ext['dow_cos'] = np.cos(2*np.pi*df_train_ext['dow']/7)

    base_cols = [target_col,'lr_resid','lr_ema10','dow_sin','dow_cos','eom'] + dynamic_columns + lowfreq_feats + tech_cols
    validate_dataframe(df_train_ext, "파생 후(원본)", base_cols)
    df_train_ext = clean_infinite_and_nan(df_train_ext, base_cols, mode="ffill_bfill_then_zero")

    # scalers
    return_scaler = StandardScaler()       # lr_resid 전용
    exogenous_scaler = StandardScaler()

    exog_cols = dynamic_columns + lowfreq_feats + ['dow_sin','dow_cos','eom'] + tech_cols
    exog_scaled = pd.DataFrame(index=df_train_ext.index)
    if exog_cols:
        exog_scaled[exog_cols] = exogenous_scaler.fit_transform(df_train_ext[exog_cols])

    # training table: [r_std, exog...]
    df_train_scaled = pd.DataFrame({'r_std': return_scaler.fit_transform(df_train_ext[['lr_resid']]).ravel()})
    for c in exog_cols: df_train_scaled[c] = exog_scaled[c]

    validate_dataframe(df_train_scaled, "스케일 후(학습 입력)", df_train_scaled.columns.tolist())
    df_train_scaled = clean_infinite_and_nan(df_train_scaled, df_train_scaled.columns.tolist(), mode="zero")

    df_target_raw = df_target[['date',target_col]].copy()

    # clamp quantiles on resid
    lr_q_low, lr_q_high = df_train_ext['lr_resid'].quantile([0.01, 0.99]).tolist()

    print(f"학습 데이터 기간: {df_train_ext['date'].min().date()} ~ {df_train_ext['date'].max().date()}")
    print(f"타깃 데이터 기간: {df_target_raw['date'].min().date()} ~ {df_target_raw['date'].max().date()}")
    print(f"동적 외생: {dynamic_columns}")
    print(f"저빈도 파생: {lowfreq_feats}")
    print(f"기술지표: {tech_cols}")
    print(f"클램프 분위수(resid): low={lr_q_low:.6e}, high={lr_q_high:.6e}")

    ema_last = float(df_train_ext['lr_ema10'].iloc[-1])

    return (df_train_scaled, df_train_ext, df_target_raw,
            return_scaler, exogenous_scaler,
            target_col, exog_cols, lr_q_low, lr_q_high, ema_last)

# ========================== sequences with policy (weekly_repeat) ==========================
def create_sequences_multistep_policy(data: pd.DataFrame, seq_length=60, pred_length=7):
    """
    훈련 시에도 예측 시와 동일한 외생 정책 사용:
    - future exog = '최근 7일 반복'(last7)  -> covariate shift 완화
    입력 data: 첫 컬럼 r_std, 이후 exog들
    """
    arr = data.values
    D = arr.shape[1]; E = D - 1
    X_seq, y_seq, X_exo_fut = [], [], []
    for i in range(seq_length, len(data) - pred_length + 1):
        past = arr[i-seq_length:i, :]              # [seq, D]
        past_exo = past[:, 1:] if E>0 else np.zeros((seq_length,0))
        last7 = past_exo[-7:] if len(past_exo) else np.zeros((7,0))
        reps = int(np.ceil(pred_length/7))
        fut_exo = np.vstack([last7]*reps)[:pred_length]  # [pred, E]
        fut_r = arr[i:i+pred_length, 0]                  # [pred] (r_std truth)
        X_seq.append(past); y_seq.append(fut_r); X_exo_fut.append(fut_exo)
    X = np.asarray(X_seq); y = np.asarray(y_seq); X_exo_future=np.asarray(X_exo_fut)
    check_array_finite("학습입력(X_seq)", X)
    check_array_finite("학습타깃(y_std)", y)
    check_array_finite("학습미래외생(X_exo_future)", X_exo_future)
    return X, y, X_exo_future

# ========================== future exog for inference ==========================
def make_future_exog_scaled_from_train(df_train_scaled: pd.DataFrame, exog_cols: List[str],
                                       horizon: int, mode="weekly_repeat") -> np.ndarray:
    if not exog_cols: return np.zeros((horizon,0), dtype=float)
    last_30 = df_train_scaled[exog_cols].iloc[-30:].to_numpy()
    if mode=="hold_last":
        fut = np.tile(last_30[-1], (horizon,1))
    elif mode=="weekly_repeat":
        last_7 = last_30[-7:]; reps=int(np.ceil(horizon/7))
        fut = np.vstack([last_7]*reps)[:horizon]
    elif mode=="zero_mean":
        fut = np.zeros((horizon, len(exog_cols)))
    else:
        raise ValueError("Unknown exog mode")
    return np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0)

# ========================== training (teacher forcing + extra losses) ==========================
def train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                epochs=220, lr=3e-4,
                                tf_start=1.0, tf_end=0.3,
                                gamma_drift=0.01, lam_sign=0.35, lam_trend=0.20, lam_slope=0.10):
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
    best=1e9; patience=25; bad=0

    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad()
        p_tf = tf_start + (tf_end - tf_start) * ((epoch-1)/(epochs-1))

        cur = Xt.clone()  # [N, seq, D]
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

        # extra losses
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

        if loss.item()<best-1e-6: best=loss.item(); bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {epoch} (best loss {best:.6f})")
                break
    print("모델 학습 완료.")
    return model

# ========================== inference ==========================
def predict_future_daily(model,
                         last_seq_scaled: np.ndarray,
                         return_scaler: StandardScaler,
                         future_exog_scaled: np.ndarray,
                         start_level: float,
                         ema_start: float,
                         r_low: float, r_high: float, tau: int = 14) -> List[float]:
    """
    입력 last_seq_scaled: [seq, 1 + E] (첫 열 r_std(resid))
    """
    device = next(model.parameters()).device
    model.eval(); H=len(future_exog_scaled)
    seq = last_seq_scaled.copy(); win = seq.shape[0]
    preds_level=[]; cur=float(start_level)

    # EMA(10) 파라미터
    alpha = 2.0 / (10.0 + 1.0)
    ema = float(ema_start)

    with torch.no_grad():
        for t in range(H):
            xt = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            endo = xt[:, :, 0:1]
            exog = xt[:, :, 1:] if seq.shape[1]>1 else xt[:, :, 0:1]
            r_std = model(endo, exog).cpu().numpy().ravel()[0]
            r_resid = return_scaler.inverse_transform([[r_std]]).ravel()[0]

            # resid clamp + 완만한 수축
            r_resid = float(np.clip(r_resid, r_low, r_high))
            r_resid *= float(np.exp(-t / max(1.0, float(tau))))

            # EMA 갱신 및 복원
            ema = alpha * (ema + r_resid) + (1 - alpha) * ema
            lr_hat = ema + r_resid

            cur = cur * np.exp(lr_hat)
            preds_level.append(cur)

            exog_t = future_exog_scaled[t] if future_exog_scaled.size else np.array([])
            r_std_next = return_scaler.transform([[r_resid]]).ravel()[0]
            new_row = np.concatenate([[r_std_next], exog_t], axis=0)
            seq = np.vstack([seq, new_row])[-win:]
    return preds_level

# ========================== plot / report ==========================
def plot_results(actual, predicted, dates, title_suffix=""):
    plt.figure(figsize=(14,7))
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, predicted, '--', label='Predicted', linewidth=2)
    plt.title(f'USD/KRW Prediction {title_suffix}'.strip())
    plt.xlabel('Date'); plt.ylabel('USD/KRW')
    plt.legend(); plt.grid(True, alpha=0.3); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig('timexer_policy_results.png', dpi=300, bbox_inches='tight'); plt.show()

def save_results(actual, predicted, dates, model_name='TimeXer (resid+EMA, attn, AR15x0.6)'):
    actual=np.asarray(actual, dtype=float); predicted=np.asarray(predicted, dtype=float)
    predicted = np.nan_to_num(predicted, nan=np.nanmean(predicted) if np.any(~np.isnan(predicted)) else 0.0)
    df=pd.DataFrame({'date':dates,'actual':actual,'predicted':predicted})
    df['error']=df['actual']-df['predicted']; df['error_pct']=df['error']/df['actual']*100.0
    df.to_csv('timexer_policy_results.csv', index=False)
    mse=mean_squared_error(actual,predicted); mae=mean_absolute_error(actual,predicted); rmse=float(np.sqrt(mse))
    dir_acc = np.mean(np.sign(actual[1:]-actual[:-1]) == np.sign(predicted[1:]-predicted[:-1]))
    print("\n예측 성능:"); print(f"MSE : {mse:.4f}"); print(f"MAE : {mae:.4f}"); print(f"RMSE: {rmse:.4f}"); print(f"Direction Acc: {dir_acc:.3f}")
    pd.DataFrame([{'MSE':mse,'MAE':mae,'RMSE':rmse,'DirAcc':dir_acc,
                   'prediction_days':len(predicted),'model':model_name}]
                 ).to_csv('timexer_policy_summary.csv', index=False)
    print("\n저장 완료:\n- timexer_policy_results.csv\n- timexer_policy_summary.csv\n- timexer_policy_results.png")

def baseline_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    # Naive-1
    naive = np.r_[actual[0], actual[:-1]]
    # Drift-EMA(10) on lr
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
    print("="*70); print("Daily policy-aligned training + path-aware losses (resid-EMA)"); print("="*70)

    (df_train_scaled, df_train_ext, df_target_raw,
     return_scaler, exogenous_scaler,
     target_col, exog_cols, lr_q_low, lr_q_high, ema_last) = load_and_preprocess_data()

    seq_len=60
    pred_len=7
    X_seq, y_std, X_exo_future = create_sequences_multistep_policy(
        df_train_scaled, seq_length=seq_len, pred_length=pred_len
    )
    print(f"학습 시퀀스: {X_seq.shape}, 타깃(r_std resid): {y_std.shape}, 미래외생(정책): {X_exo_future.shape}")

    # model
    D = df_train_scaled.shape[1]; exog_dim = max(1, D-1)
    if exog_dim == 1 and D <= 1:
        X_seq = np.concatenate([X_seq, np.zeros((X_seq.shape[0], X_seq.shape[1], 1), dtype=X_seq.dtype)], axis=2)
        X_exo_future = np.zeros((X_exo_future.shape[0], X_exo_future.shape[1], 1), dtype=X_exo_future.dtype)
        exog_dim = 1

    model = SimpleTimeXerModel(endogenous_dim=1, exogenous_dim=exog_dim,
                               hidden_size=128, num_layers=2, ar_lags=15, ar_gain=0.60)
    model = train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                        epochs=220, lr=3e-4, tf_start=1.0, tf_end=0.3,
                                        gamma_drift=0.01, lam_sign=0.35, lam_trend=0.20, lam_slope=0.10)

    # inference
    last_seq_scaled = df_train_scaled.iloc[-seq_len:].to_numpy()
    check_array_finite("마지막시퀀스(last_seq_scaled)", last_seq_scaled)

    H = len(df_target_raw)
    future_exog_scaled = make_future_exog_scaled_from_train(df_train_scaled, exog_cols, H, mode="weekly_repeat")
    check_array_finite("타깃외생(future_exog_scaled)", future_exog_scaled)

    start_level = float(df_train_ext[target_col].iloc[-1])
    preds = predict_future_daily(model, last_seq_scaled, return_scaler,
                                 future_exog_scaled, start_level,
                                 ema_start=ema_last,
                                 r_low=lr_q_low, r_high=lr_q_high, tau=14)

    actual_values = df_target_raw[target_col].to_numpy()
    actual_dates = df_target_raw['date']
    print("\n예측 결과 요약:"); print(f"실제값 범위: {actual_values.min():.1f} ~ {actual_values.max():.1f}")
    print(f"예측값 범위: {np.nanmin(preds):.1f} ~ {np.nanmax(preds):.1f}")

    plot_results(actual_values, preds, actual_dates,
                 title_suffix=f"({actual_dates.min().date()} ~ {actual_dates.max().date()})")
    save_results(actual_values, preds, actual_dates)
    baseline_metrics(actual_values, preds)

    print("\n"+"="*70); print("완료"); print("="*70)

if __name__ == "__main__":
    main()

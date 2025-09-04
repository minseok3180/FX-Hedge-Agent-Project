#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USD/KRW daily forecasting — LR(target) 버전 (직선 하방편향 해소)
- Target: lr = log(y_t / y_{t-1})  → 표준화(r_std) 후 학습
- Restore: y_t = y_{t-1} * exp(lr_hat)  (EMA 복원 불필요)
- Model: LSTM(endo/exo) + cross-attn + MLP + linear AR head(lags=15, gain=0.6)
- Train: multi-step teacher forcing(예측시 외생 정책과 동일), 추가 손실(trend/slope/sign/drift)
- Exog: lr_ema10(추세), 캘린더(dow_sin, dow_cos, eom), 기술지표, + r_std 래그(15개; 항상 포함)
- Inference: lr 분위수 클램프(1~99%), 기본 수축 없음(원하면 shrink_tau 크게)
"""

import warnings, numpy as np, pandas as pd
from typing import List
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


# ========================== data load / preprocess (DAILY-ONLY) ==========================
def load_and_preprocess_data():
    print("데이터 로드 중...")
    df_train = pd.read_csv('df.csv'); df_train['date']=pd.to_datetime(df_train['date'])
    df_train = df_train.sort_values('date').reset_index(drop=True)

    df_target = pd.read_csv('df_target.csv'); df_target['date']=pd.to_datetime(df_target['date'])
    df_target = df_target.sort_values('date').reset_index(drop=True)

    # 저빈도 외생 제거(존재시)
    drop_if_exists=['base','us_current','us_growth','us_interest',
                    'us_ex','us_im','reserve','us_reserve','us_export','us_import',
                    'us_gdp','consumer','exp_rate','im_rate','us_stock','dir_inv','us_indpro','us_unemp','us_prod']
    drop_real=[c for c in drop_if_exists if c in df_train.columns]
    if drop_real:
        df_train=df_train.drop(columns=drop_real)
        df_target=df_target.drop(columns=[c for c in drop_real if c in df_target.columns], errors='ignore')

    # 일별 동적 변수만 사용(존재하는 컬럼만)
    dynamic_columns=[c for c in ['market'] if c in df_train.columns]

    target_col='usdkrw(target)'
    y = df_train[target_col].astype(float)

    # --- log-return 타깃 ---
    lr = np.log(y / y.shift(1)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    df_train['lr'] = lr
    df_train['lr_ema10'] = lr.ewm(span=10, adjust=False).mean()

    # 기술지표(레벨/수익률 혼합)
    ma20 = y.rolling(20).mean(); std20 = y.rolling(20).std()
    df_train['lvl_z20']   = (y - ma20) / (std20 + 1e-12)
    df_train['lvl_rsi14'] = rsi(y, 14) / 100.0
    df_train['lr_ma5']    = lr.rolling(5).mean()
    df_train['lr_ma10']   = lr.rolling(10).mean()
    df_train['lr_vol10']  = lr.rolling(10).std()

    tech_cols=['lvl_z20','lvl_rsi14','lr_ma5','lr_ma10','lr_vol10']

    # calendar
    df_train['dow']     = df_train['date'].dt.weekday
    df_train['eom']     = (df_train['date'].dt.is_month_end).astype(int)
    df_train['dow_sin'] = np.sin(2*np.pi*df_train['dow']/7)
    df_train['dow_cos'] = np.cos(2*np.pi*df_train['dow']/7)

    base_cols = [target_col,'lr','lr_ema10','dow_sin','dow_cos','eom'] + dynamic_columns + tech_cols
    validate_dataframe(df_train, "파생 후(원본, 일별만)", base_cols)
    df_train = clean_infinite_and_nan(df_train, base_cols, mode="ffill_bfill_then_zero")

    # 스케일러
    return_scaler   = StandardScaler()     # lr 전용
    exogenous_scaler= StandardScaler()

    # r_std 생성(타깃)
    r_std = return_scaler.fit_transform(df_train[['lr']]).ravel()
    df_train_scaled = pd.DataFrame({'r_std': r_std})

    # exog 스케일
    exog_cols_core = dynamic_columns + ['lr_ema10','dow_sin','dow_cos','eom'] + tech_cols
    exog_scaled = pd.DataFrame(index=df_train.index)
    if exog_cols_core:
        exog_scaled[exog_cols_core] = exogenous_scaler.fit_transform(df_train[exog_cols_core])

    # === r_std 래그 15개를 항상 exogenous로 추가 ===
    LAG_K = 15
    for k in range(1, LAG_K+1):
        df_train_scaled[f'rstd_lag{k}'] = df_train_scaled['r_std'].shift(k)

    df_train_scaled = df_train_scaled.join(exog_scaled)
    df_train_scaled = df_train_scaled.fillna(0.0)

    exog_cols = [c for c in df_train_scaled.columns if c != 'r_std']

    df_target_raw = df_target[['date',target_col]].copy()

    # lr 클램프 분위수
    lr_q_low, lr_q_high = df_train['lr'].quantile([0.01, 0.99]).tolist()

    print(f"학습 기간: {df_train['date'].min().date()} ~ {df_train['date'].max().date()}")
    print(f"예측 기간: {df_target_raw['date'].min().date()} ~ {df_target_raw['date'].max().date()}")
    print(f"동적 외생(일별): {dynamic_columns}")
    print(f"기술지표(일별): {tech_cols}")
    print(f"exog 채널 수(래그 포함): {len(exog_cols)}")
    print(f"클램프 분위수(lr): low={lr_q_low:.3e}, high={lr_q_high:.3e}")

    start_level = float(df_train[target_col].iloc[-1])

    return (df_train_scaled, df_target_raw, return_scaler, exogenous_scaler,
            target_col, exog_cols, lr_q_low, lr_q_high, start_level)


# ========================== sequences with policy (weekly_repeat) ==========================
def create_sequences_multistep_policy(data: pd.DataFrame, seq_length=60, pred_length=7):
    """
    학습 시에도 예측 시와 동일한 외생 정책 사용:
    - future exog = '최근 7일 반복'(last7)
    입력: 첫 컬럼 r_std, 이후 exog들(여기엔 r_std 래그 포함)
    """
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
                                       horizon: int, mode="weekly_repeat") -> np.ndarray:
    if not exog_cols:
        return np.zeros((horizon,0), dtype=float)
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


# ========================== training ==========================
def train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                epochs=220, lr=3e-4,
                                tf_start=1.0, tf_end=0.2,
                                gamma_drift=0.01, lam_sign=0.25, lam_trend=0.45, lam_slope=0.20,
                                patience=40):
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

        # 누적 로그수익률 일치(추세), 기울기, 방향성, 드리프트 억제
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
                         lr_low: float, lr_high: float,
                         shrink_tau: float = None) -> List[float]:
    """
    입력 last_seq_scaled: [seq, 1 + E] (첫 열 r_std=표준화된 lr)
    """
    device = next(model.parameters()).device
    model.eval(); H=len(future_exog_scaled)
    seq = last_seq_scaled.copy(); win = seq.shape[0]
    preds_level=[]; cur=float(start_level)

    with torch.no_grad():
        for t in range(H):
            xt = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            endo = xt[:, :, 0:1]
            exog = xt[:, :, 1:] if seq.shape[1]>1 else xt[:, :, 0:1]
            r_std = model(endo, exog).cpu().numpy().ravel()[0]
            lr_hat = return_scaler.inverse_transform([[r_std]]).ravel()[0]

            # clamp + optional shrink
            lr_hat = float(np.clip(lr_hat, lr_low, lr_high))
            if shrink_tau is not None and shrink_tau > 0:
                lr_hat *= float(np.exp(-t / float(shrink_tau)))  # 매우 큰 값(예: 90) 권장

            cur = cur * np.exp(lr_hat)
            preds_level.append(cur)

            exog_t = future_exog_scaled[t] if future_exog_scaled.size else np.array([])
            r_std_next = return_scaler.transform([[lr_hat]]).ravel()[0]
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
    plt.tight_layout(); plt.savefig('timexer_lr_results.png', dpi=300, bbox_inches='tight'); plt.show()

def save_results(actual, predicted, dates, model_name='TimeXer (LR target, attn, AR15x0.6, lag-exog)'):
    actual=np.asarray(actual, dtype=float); predicted=np.asarray(predicted, dtype=float)
    predicted = np.nan_to_num(predicted, nan=np.nanmean(predicted) if np.any(~np.isnan(predicted)) else 0.0)
    df=pd.DataFrame({'date':dates,'actual':actual,'predicted':predicted})
    df['error']=df['actual']-df['predicted']; df['error_pct']=df['error']/df['actual']*100.0
    df.to_csv('timexer_lr_results.csv', index=False)
    mse=mean_squared_error(actual,predicted); mae=mean_absolute_error(actual,predicted); rmse=float(np.sqrt(mse))
    dir_acc = np.mean(np.sign(actual[1:]-actual[:-1]) == np.sign(predicted[1:]-predicted[:-1]))
    print("\n예측 성능:"); print(f"MSE : {mse:.4f}"); print(f"MAE : {mae:.4f}"); print(f"RMSE: {rmse:.4f}"); print(f"Direction Acc: {dir_acc:.3f}")
    pd.DataFrame([{'MSE':mse,'MAE':mae,'RMSE':rmse,'DirAcc':dir_acc,
                   'prediction_days':len(predicted),'model':model_name}]
                 ).to_csv('timexer_lr_summary.csv', index=False)
    print("\n저장 완료:\n- timexer_lr_results.csv\n- timexer_lr_summary.csv\n- timexer_lr_results.png")

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
    print("="*70); print("Daily-only exog (LR target; policy-aligned; lag-exog)"); print("="*70)

    (df_train_scaled, df_target_raw, return_scaler, exogenous_scaler,
     target_col, exog_cols, lr_q_low, lr_q_high, start_level) = load_and_preprocess_data()

    seq_len=90          # 추세 강화(기존 60→90 권장)
    pred_len=7
    X_seq, y_std, X_exo_future = create_sequences_multistep_policy(
        df_train_scaled, seq_length=seq_len, pred_length=pred_len
    )
    print(f"학습 시퀀스: {X_seq.shape}, 타깃(r_std=lr_std): {y_std.shape}, 미래외생: {X_exo_future.shape}")

    D = df_train_scaled.shape[1]; exog_dim = max(1, D-1)
    model = SimpleTimeXerModel(endogenous_dim=1, exogenous_dim=exog_dim,
                               hidden_size=128, num_layers=2, ar_lags=15, ar_gain=0.60)
    model = train_model_teacher_forcing(model, X_seq, y_std, X_exo_future,
                                        epochs=240, lr=3e-4, tf_start=1.0, tf_end=0.2,
                                        gamma_drift=0.01, lam_sign=0.25, lam_trend=0.45, lam_slope=0.20,
                                        patience=45)

    last_seq_scaled = df_train_scaled.iloc[-seq_len:].to_numpy()
    check_array_finite("마지막시퀀스(last_seq_scaled)", last_seq_scaled)

    H = len(df_target_raw)
    future_exog_scaled = make_future_exog_scaled_from_train(df_train_scaled, exog_cols, H, mode="weekly_repeat")
    check_array_finite("타깃외생(future_exog_scaled)", future_exog_scaled)

    preds = predict_future_daily(model, last_seq_scaled, return_scaler,
                                 future_exog_scaled, start_level=start_level,
                                 lr_low=lr_q_low, lr_high=lr_q_high,
                                 shrink_tau=None)  # 수축 제거

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

#!/usr/bin/env python3
"""
df.csv로 간단한 TimeXer 스타일 모델을 학습하고 df_target.csv를 예측하는 스크립트
- 비어있는 칼럼(us_current, us_growth, us_interest)은 제거
- 월별/연별 저빈도 변수는 제거하지 않고 변화율(MoM/YoY) 파생변수를 사용
- df_target.csv에는 usdkrw(target)만 있어도 실행 가능(외생은 코드에서 간단 예측으로 생성)
- NaN/Inf 안전장치 및 상세 검증 로그 포함
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =======================================================================================
# 0) 유틸: 검증/정리/안전 pct_change
# =======================================================================================

def validate_dataframe(df: pd.DataFrame, name: str, cols: list):
    bad = {}
    for c in cols:
        s = df[c]
        n_nan = int(s.isna().sum())
        n_inf = int(np.isinf(s).sum())
        if n_nan or n_inf:
            bad[c] = (n_nan, n_inf)
    if bad:
        print(f"[검증 경고] {name}에 비정상 값 발견:")
        for c, (nn, ni) in bad.items():
            print(f"  - {c}: NaN={nn}, Inf={ni}")
    else:
        print(f"[검증 OK] {name}: NaN/Inf 없음")

def clean_infinite_and_nan(df: pd.DataFrame, cols: list, mode: str = "zero") -> pd.DataFrame:
    # Inf → NaN
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    if mode == "ffill_bfill_then_zero":
        df[cols] = df[cols].ffill().bfill().fillna(0.0)
    elif mode == "zero":
        df[cols] = df[cols].fillna(0.0)
    else:
        raise ValueError("mode must be 'zero' or 'ffill_bfill_then_zero'")
    return df

def _pct_change_safe(s: pd.Series, periods: int) -> pd.Series:
    prev = s.shift(periods)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (s - prev) / prev
    pct = pct.replace([np.inf, -np.inf], np.nan)
    return pct

def check_array_finite(name: str, arr: np.ndarray):
    n = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"[배열검증] {name}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf}, valid={(n - n_nan - n_inf)}/{n}")

# =======================================================================================
# 1) 모델
# =======================================================================================

class SimpleTimeXerModel(nn.Module):
    def __init__(self, endogenous_dim=1, exogenous_dim=5, hidden_size=128, num_layers=2, output_size=7):
        super().__init__()
        self.endogenous_lstm = nn.LSTM(
            input_size=endogenous_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.1
        )
        self.exogenous_lstm = nn.LSTM(
            input_size=exogenous_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.1
        )
        self.global_token = nn.Parameter(torch.randn(1, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, endogenous_x, exogenous_x):
        # endogenous_x: [B, T, 1], exogenous_x: [B, T, E]
        B = endogenous_x.size(0)
        end_out, _ = self.endogenous_lstm(endogenous_x)
        exo_out, _ = self.exogenous_lstm(exogenous_x)
        end_feat = end_out[:, -1, :]   # [B, H]
        exo_feat = exo_out[:, -1, :]   # [B, H]
        attn_out, _ = self.cross_attention(
            query=end_feat.unsqueeze(1),
            key=exo_feat.unsqueeze(1),
            value=exo_feat.unsqueeze(1)
        )
        attn_out = attn_out.squeeze(1)  # [B, H]
        global_tokens = self.global_token.repeat(B, 1)  # [B, H]
        feat = torch.cat([attn_out, global_tokens], dim=1)  # [B, 2H]
        yhat = self.output_layer(feat)  # [B, pred_len]
        return yhat

# =======================================================================================
# 2) 저빈도 변수 파생: MoM/YoY 유틸
# =======================================================================================

def add_lowfreq_change_features(
    df_daily: pd.DataFrame,
    date_col: str,
    monthly_cols: List[str],
    annual_cols: List[str],
    prefix_m: str = "mom_",
    prefix_y: str = "yoy_",
    anchor: str = "start",            # 'start' or 'release'
    release_lag_days: int = 0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    월별/연별 저빈도 컬럼을 전월/전년 대비 %변화율로 파생하고, 일단위로 ffill하여 붙임.
    반환: (df, 생성된 파생컬럼 리스트)
    """
    df = df_daily.sort_values(date_col).copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    gen_cols: List[str] = []

    # 월별: 전월 대비
    if monthly_cols:
        monthly_df = (
            df[[date_col, "year", "month"] + monthly_cols]
            .drop_duplicates(subset=["year", "month"])
            .groupby(["year", "month"], as_index=False).first()
            .sort_values(["year", "month"])
            .reset_index(drop=True)
        )
        monthly_df["anchor_date"] = pd.to_datetime(
            dict(year=monthly_df["year"], month=monthly_df["month"], day=1)
        )
        if anchor == "release":
            monthly_df["anchor_date"] = monthly_df["anchor_date"] + pd.to_timedelta(release_lag_days, unit="D")

        mom_blocks = []
        for col in monthly_cols:
            series = monthly_df[col].astype(float)
            mom = _pct_change_safe(series, periods=1).fillna(0.0)  # 첫 달 0
            mom_blocks.append(pd.DataFrame({"anchor_date": monthly_df["anchor_date"], f"{prefix_m}{col}": mom}))
            gen_cols.append(f"{prefix_m}{col}")

        mom_monthly = mom_blocks[0]
        for blk in mom_blocks[1:]:
            mom_monthly = mom_monthly.merge(blk, on="anchor_date", how="outer")
        mom_monthly = mom_monthly.sort_values("anchor_date")

        df = df.merge(mom_monthly.rename(columns={"anchor_date": date_col}), on=date_col, how="left")
        df[gen_cols] = df[gen_cols].ffill()
        df[gen_cols] = df[gen_cols].bfill()
        df[gen_cols] = df[gen_cols].fillna(0.0)

    # 연별: 전년 대비
    if annual_cols:
        yearly_df = (
            df[[date_col, "year"] + annual_cols]
            .drop_duplicates(subset=["year"])
            .groupby(["year"], as_index=False).first()
            .sort_values(["year"])
            .reset_index(drop=True)
        )
        yearly_df["anchor_date"] = pd.to_datetime(dict(year=yearly_df["year"], month=1, day=1))
        if anchor == "release":
            yearly_df["anchor_date"] = yearly_df["anchor_date"] + pd.to_timedelta(release_lag_days, unit="D")

        yoy_blocks = []
        for col in annual_cols:
            series = yearly_df[col].astype(float)
            yoy = _pct_change_safe(series, periods=1).fillna(0.0)
            yoy_blocks.append(pd.DataFrame({"anchor_date": yearly_df["anchor_date"], f"{prefix_y}{col}": yoy}))
            gen_cols.append(f"{prefix_y}{col}")

        yoy_yearly = yoy_blocks[0]
        for blk in yoy_blocks[1:]:
            yoy_yearly = yoy_yearly.merge(blk, on="anchor_date", how="outer")
        yoy_yearly = yoy_yearly.sort_values("anchor_date")

        df = df.merge(yoy_yearly.rename(columns={"anchor_date": date_col}), on=date_col, how="left")
        year_cols = [c for c in gen_cols if c.startswith(prefix_y)]
        df[year_cols] = df[year_cols].ffill()
        df[year_cols] = df[year_cols].bfill()
        df[year_cols] = df[year_cols].fillna(0.0)

    df = df.drop(columns=["year", "month"], errors="ignore")
    return df, gen_cols

# =======================================================================================
# 3) 데이터 로드 및 전처리
# =======================================================================================

def load_and_preprocess_data():
    print("데이터 로드 중...")
    df_train = pd.read_csv('df.csv')
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train = df_train.sort_values('date').reset_index(drop=True)

    df_target = pd.read_csv('df_target.csv')
    df_target['date'] = pd.to_datetime(df_target['date'])
    df_target = df_target.sort_values('date').reset_index(drop=True)

    # 제거 대상(비어 있는 칼럼들)
    drop_if_exists = ['us_current', 'us_growth', 'us_interest']
    drop_real = [c for c in drop_if_exists if c in df_train.columns]
    if drop_real:
        df_train = df_train.drop(columns=drop_real)
        df_target = df_target.drop(columns=[c for c in drop_real if c in df_target.columns], errors='ignore')

    # 동적(일별 변동) 변수(예시)
    dynamic_columns = ['base', 'market', 'consumer', 'exp_rate', 'im_rate']
    dynamic_columns = [c for c in dynamic_columns if c in df_train.columns]

    # 저빈도 컬럼 후보
    monthly_like = [
        'us_ex', 'us_im', 'reserve', 'us_reserve',
        'us_export', 'us_import', 'us_gdp', 'us_stock'
    ]
    monthly_like = [c for c in monthly_like if c in df_train.columns]

    # 연별 컬럼(필요시 추가)
    annual_like: List[str] = []

    # 저빈도 → MoM/YoY 파생 추가
    df_train_ext, lowfreq_feats = add_lowfreq_change_features(
        df_daily=df_train,
        date_col="date",
        monthly_cols=monthly_like,
        annual_cols=annual_like,
        prefix_m="mom_",
        prefix_y="yoy_",
        anchor="start",
        release_lag_days=0
    )

    # 최종 학습 피처: 타깃 + 동적 + 저빈도 파생
    target_col = 'usdkrw(target)'
    feature_columns = [target_col] + dynamic_columns + lowfreq_feats

    # ---- 스케일 전 검증/정리 ----
    if feature_columns:
        validate_dataframe(df_train_ext, "파생 후(원본 스케일)", feature_columns)
        # 동적 컬럼에 누락이 있을 수 있어 ffill/bfill 후 0
        df_train_ext = clean_infinite_and_nan(df_train_ext, feature_columns, mode="ffill_bfill_then_zero")

    # 스케일러
    usdkrw_scaler = StandardScaler()
    exogenous_scaler = StandardScaler()

    # 학습 스케일링
    df_train_scaled = df_train_ext[feature_columns].copy()
    df_train_scaled[target_col] = usdkrw_scaler.fit_transform(df_train_ext[[target_col]])
    exog_cols = [c for c in feature_columns if c != target_col]
    if exog_cols:
        df_train_scaled[exog_cols] = exogenous_scaler.fit_transform(df_train_ext[exog_cols])

    # ---- 스케일 후 검증/정리 ----
    validate_dataframe(df_train_scaled, "스케일 후(학습 입력)", df_train_scaled.columns.tolist())
    df_train_scaled = clean_infinite_and_nan(df_train_scaled, df_train_scaled.columns.tolist(), mode="zero")

    # 타깃 데이터(정답)
    df_target_raw = df_target[['date', target_col]].copy()

    print(f"학습 데이터 기간: {df_train['date'].min().date()} ~ {df_train['date'].max().date()}")
    print(f"타깃 데이터 기간: {df_target_raw['date'].min().date()} ~ {df_target_raw['date'].max().date()}")
    print(f"동적 외생: {dynamic_columns}")
    print(f"저빈도 파생: {lowfreq_feats}")

    return (
        df_train_scaled, df_train_ext, df_target_raw,
        usdkrw_scaler, exogenous_scaler,
        target_col, exog_cols
    )

def create_sequences(data: pd.DataFrame, seq_length=30, pred_length=7, target_col='usdkrw(target)'):
    X, y = [], []
    cols = data.columns.tolist()
    arr = data.values  # [N, D]
    for i in range(seq_length, len(data) - pred_length + 1):
        seq = arr[i-seq_length:i]                # [T, D]
        tgt = arr[i:i+pred_length, cols.index(target_col)]  # [pred_len]
        X.append(seq)
        y.append(tgt)
    X = np.array(X)
    y = np.array(y)
    # 배열 유효성 검사
    check_array_finite("학습입력(X)", X)
    check_array_finite("학습타깃(y)", y)
    return X, y

# =======================================================================================
# 4) 외생 생성(타깃 구간): 간단 예측 정책
# =======================================================================================

def make_future_exog_scaled_from_train(
    df_train_scaled: pd.DataFrame,
    exog_cols: List[str],
    horizon: int,
    mode: str = "hold_last"  # 'hold_last' | 'weekly_repeat' | 'zero_mean'
) -> np.ndarray:
    if not exog_cols:
        return np.zeros((horizon, 0), dtype=float)

    last_30 = df_train_scaled[exog_cols].iloc[-30:].to_numpy()  # [30, E]

    if mode == "hold_last":
        last_vec = last_30[-1]                # [E]
        future_exog = np.tile(last_vec, (horizon, 1))
    elif mode == "weekly_repeat":
        last_7 = last_30[-7:]                 # [7, E]
        reps = int(np.ceil(horizon / 7))
        future_exog = np.vstack([last_7] * reps)[:horizon]
    elif mode == "zero_mean":
        future_exog = np.zeros((horizon, len(exog_cols)), dtype=float)
    else:
        raise ValueError(f"Unknown exog mode: {mode}")

    # 방어
    future_exog = np.nan_to_num(future_exog, nan=0.0, posinf=0.0, neginf=0.0)
    return future_exog

# =======================================================================================
# 5) 학습/예측/평가
# =======================================================================================

def train_model(model, train_sequences, train_targets, epochs=100, learning_rate=5e-4):
    print("모델 학습 시작...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    model = model.to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.tensor(train_sequences, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(train_targets, dtype=torch.float32, device=device)

    # 입력 유효성 빠른 체크
    if not torch.isfinite(X_tensor).all():
        print("[학습 전 경고] X_tensor에 NaN/Inf 존재 — 정리 필요")
    if not torch.isfinite(y_tensor).all():
        print("[학습 전 경고] y_tensor에 NaN/Inf 존재 — 정리 필요")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        endo = X_tensor[:, :, 0:1]
        exog = X_tensor[:, :, 1:]
        out = model(endo, exog)

        if not torch.isfinite(out).all():
            print("[학습 경고] 모델 출력에 NaN/Inf 발생 — 0으로 대체")
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        loss = criterion(out, y_tensor)
        if not torch.isfinite(loss):
            print("[학습 경고] 손실이 NaN/Inf — 에폭 중단")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}")
    print("모델 학습 완료.")
    return model

def predict_future(
    model,
    last_sequence_scaled: np.ndarray,     # [30, 1+E] (학습 스케일)
    usdkrw_scaler: StandardScaler,
    future_exog_scaled: np.ndarray,       # [H, E] (학습 스케일)
    pred_length: int = 7
):
    device = next(model.parameters()).device
    model.eval()
    H = len(future_exog_scaled)
    preds_std = []
    current = last_sequence_scaled.copy()  # [30, 1+E]

    with torch.no_grad():
        t = 0
        while t < H:
            seq = torch.tensor(current, dtype=torch.float32, device=device).unsqueeze(0)
            endo = seq[:, :, 0:1]
            exog = seq[:, :, 1:]
            yhat_vec = model(endo, exog).squeeze(0).detach().cpu().numpy()

            # 방어: 출력 NaN/Inf
            if not np.isfinite(yhat_vec).all():
                print("[추론 경고] 예측 벡터에 NaN/Inf — 0으로 대체")
                yhat_vec = np.nan_to_num(yhat_vec, nan=0.0, posinf=0.0, neginf=0.0)

            for j in range(pred_length):
                if t >= H:
                    break
                yhat_t = yhat_vec[j]
                exog_t = future_exog_scaled[t]
                if not np.isfinite(exog_t).all():
                    exog_t = np.nan_to_num(exog_t, nan=0.0, posinf=0.0, neginf=0.0)

                new_row = np.concatenate([[yhat_t], exog_t], axis=0)
                current = np.vstack([current, new_row])
                if len(current) > 30:
                    current = current[-30:]
                preds_std.append(yhat_t)
                t += 1

    preds_std_arr = np.array(preds_std).reshape(-1, 1)
    preds_rescaled = usdkrw_scaler.inverse_transform(preds_std_arr).ravel().tolist()
    # 방어: 역정규화 후도 NaN 방지
    preds_rescaled = np.nan_to_num(preds_rescaled, nan=float(np.nanmean(preds_rescaled) if len(preds_rescaled) else 0.0)).tolist()
    return preds_rescaled

def plot_results(actual, predicted, dates, title_suffix=""):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, predicted, '--', label='Predicted', linewidth=2)
    plt.title(f'USD/KRW Prediction {title_suffix}'.strip())
    plt.xlabel('Date')
    plt.ylabel('USD/KRW')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lstm_timexer_style_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(actual, predicted, dates, model_name='Simple TimeXer'):
    # 방어: 성능계산 전에 NaN 제거/대체
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not np.isfinite(predicted).all():
        print("[평가 경고] 예측에 NaN/Inf — 0으로 대체 후 평가")
        predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)

    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predicted,
        'error': actual - predicted,
        'error_pct': (actual - predicted) / actual * 100.0
    })
    results_df.to_csv('lstm_timexer_style_prediction_results.csv', index=False)

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    print("\n예측 성능:")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    pd.DataFrame([{
        'MSE': mse, 'MAE': mae, 'RMSE': rmse,
        'prediction_days': len(predicted), 'model': model_name
    }]).to_csv('lstm_timexer_style_summary.csv', index=False)

    print("\n저장 완료:")
    print("- lstm_timexer_style_prediction_results.csv")
    print("- lstm_timexer_style_summary.csv")
    print("- lstm_timexer_style_prediction_results.png")

# =======================================================================================
# 6) 메인
# =======================================================================================

def main():
    print("=" * 70)
    print("LSTM-based Model with TimeXer Ideas: Training on df.csv, Predicting df_target.csv")
    print("=" * 70)

    (df_train_scaled, df_train_ext, df_target_raw,
     usdkrw_scaler, exogenous_scaler,
     target_col, exog_cols) = load_and_preprocess_data()

    # 학습 시퀀스
    seq_length = 30
    pred_length = 7
    X, y = create_sequences(df_train_scaled, seq_length, pred_length, target_col=target_col)
    print(f"학습 시퀀스: {X.shape}, 타깃: {y.shape}")

    # 모델
    exog_dim = len(exog_cols)
    if exog_dim == 0:
        exog_dim_for_model = 1
        X = np.concatenate([X[:, :, :1], np.zeros((X.shape[0], X.shape[1], 1), dtype=X.dtype)], axis=2)
    else:
        exog_dim_for_model = exog_dim

    model = SimpleTimeXerModel(
        endogenous_dim=1, exogenous_dim=exog_dim_for_model,
        hidden_size=128, num_layers=2, output_size=pred_length
    )
    model = train_model(model, X, y, epochs=100, learning_rate=5e-4)

    # 예측 준비: 마지막 시퀀스
    last_seq_scaled = df_train_scaled.iloc[-seq_length:].to_numpy()  # [30, 1+E]
    if exog_dim == 0:
        last_seq_scaled = np.concatenate([last_seq_scaled[:, :1], np.zeros((seq_length, 1))], axis=1)
    check_array_finite("마지막시퀀스(last_seq_scaled)", last_seq_scaled)

    # 타깃 구간 외생을 간단 예측으로 생성
    H = len(df_target_raw)
    exog_mode = "hold_last"  # 'hold_last' | 'weekly_repeat' | 'zero_mean'
    future_exog_scaled = make_future_exog_scaled_from_train(
        df_train_scaled=df_train_scaled,
        exog_cols=exog_cols,
        horizon=H,
        mode=exog_mode
    )
    if exog_dim == 0:
        future_exog_scaled = np.zeros((H, 1), dtype=float)
    check_array_finite("타깃외생(future_exog_scaled)", future_exog_scaled)

    # 예측
    preds = predict_future(
        model=model,
        last_sequence_scaled=last_seq_scaled,
        usdkrw_scaler=usdkrw_scaler,
        future_exog_scaled=future_exog_scaled,
        pred_length=pred_length
    )

    # 결과
    actual_values = df_target_raw[target_col].to_numpy()
    actual_dates = df_target_raw['date']
    print("\n예측 결과 요약:")
    print(f"실제값 범위: {actual_values.min():.1f} ~ {actual_values.max():.1f}")
    print(f"예측값 범위: {np.nanmin(preds):.1f} ~ {np.nanmax(preds):.1f}")

    plot_results(actual_values, preds, actual_dates,
                 title_suffix=f"({actual_dates.min().date()} ~ {actual_dates.max().date()})")
    save_results(actual_values, preds, actual_dates, model_name='Simple TimeXer')

    print("\n" + "=" * 70)
    print("완료")
    print("=" * 70)

if __name__ == "__main__":
    main()

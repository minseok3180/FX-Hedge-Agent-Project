#!/usr/bin/env python3
"""
df_trial.csv로 간단한 TimeXer 스타일 모델을 학습하고 target_trial.csv를 예측하는 스크립트
TimeXer의 핵심 아이디어: 내생변수와 외생변수를 분리하여 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 간단한 TimeXer 스타일 모델 구현
# ============================================================================

class SimpleTimeXerModel(nn.Module):
    """
    TimeXer의 핵심 아이디어를 적용한 간단한 모델:
    - 내생변수(usdkrw): LSTM으로 시간적 패턴 학습
    - 외생변수: 별도 처리 후 결합
    - 글로벌 토큰: 두 정보를 연결하는 브릿지
    """
    
    def __init__(self, endogenous_dim=1, exogenous_dim=5, hidden_size=128, num_layers=2, output_size=7):
        super(SimpleTimeXerModel, self).__init__()
        
        self.endogenous_dim = endogenous_dim
        self.exogenous_dim = exogenous_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 내생변수 처리 (usdkrw)
        self.endogenous_lstm = nn.LSTM(
            input_size=endogenous_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 외생변수 처리
        self.exogenous_lstm = nn.LSTM(
            input_size=exogenous_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 글로벌 토큰 (TimeXer의 핵심 아이디어)
        self.global_token = nn.Parameter(torch.randn(1, hidden_size))
        
        # 교차 주의 메커니즘 (Cross-Attention)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, endogenous_x, exogenous_x):
        """
        Args:
            endogenous_x: [batch_size, seq_len, 1] - usdkrw 데이터
            exogenous_x: [batch_size, seq_len, 5] - 외생변수 데이터
        """
        batch_size = endogenous_x.size(0)
        
        # 1. 내생변수 처리 (TimeXer의 EnEmbedding 역할)
        endogenous_out, (endogenous_h, _) = self.endogenous_lstm(endogenous_x)
        endogenous_features = endogenous_out[:, -1, :]  # 마지막 시점의 특징
        
        # 2. 외생변수 처리 (TimeXer의 DataEmbedding_inverted 역할)
        exogenous_out, (exogenous_h, _) = self.exogenous_lstm(exogenous_x)
        exogenous_features = exogenous_out[:, -1, :]  # 마지막 시점의 특징
        
        # 3. 글로벌 토큰 적용 (TimeXer의 핵심 아이디어)
        global_tokens = self.global_token.repeat(batch_size, 1)
        
        # 4. 교차 주의를 통한 정보 결합 (TimeXer의 Cross-Attention 역할)
        # endogenous_features를 query로, exogenous_features를 key, value로 사용
        cross_features, _ = self.cross_attention(
            query=endogenous_features.unsqueeze(1),
            key=exogenous_features.unsqueeze(1),
            value=exogenous_features.unsqueeze(1)
        )
        cross_features = cross_features.squeeze(1)
        
        # 5. 글로벌 토큰과 결합
        combined_features = torch.cat([cross_features, global_tokens], dim=1)
        
        # 6. 최종 출력
        output = self.output_layer(combined_features)
        
        return output

# ============================================================================
# 2. 데이터 전처리 및 시퀀스 생성
# ============================================================================

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    print("📊 데이터 로드 중...")
    
    # 학습 데이터 로드
    df_train = pd.read_csv('df_trial.csv')
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train = df_train.sort_values('date').reset_index(drop=True)
    
    # 예측 대상 데이터 로드
    df_target = pd.read_csv('target_trial.csv')
    df_target['date'] = pd.to_datetime(df_target['date'])
    df_target = df_target.sort_values('date').reset_index(drop=True)
    
    print(f"학습 데이터: {df_train.shape} (2023-01-01 ~ 2024-12-31)")
    print(f"예측 대상: {df_target.shape} (2025-01-01 ~ 2025-01-31)")
    
    # 외생변수 중 변동이 적은 것들 제거 (월별 데이터)
    # 변동이 적은 컬럼들: us_ex, us_im, reserve, us_reserve, us_export, us_import, us_gdp, us_stock
    # 변동이 있는 컬럼들: base, market, consumer, exp_rate, im_rate
    static_columns = ['us_ex', 'us_im', 'reserve', 'us_reserve', 'us_export', 'us_import', 'us_gdp', 'us_stock']
    dynamic_columns = ['base', 'market', 'consumer', 'exp_rate', 'im_rate']
    
    print(f"제거할 정적 외생변수: {static_columns}")
    print(f"사용할 동적 외생변수: {dynamic_columns}")
    
    # 최종 사용할 컬럼들
    feature_columns = ['usdkrw(target)'] + dynamic_columns
    print(f"최종 특성 컬럼: {feature_columns}")
    
    # usdkrw와 외생변수를 별도로 정규화
    usdkrw_scaler = StandardScaler()
    exogenous_scaler = StandardScaler()
    
    # usdkrw 정규화
    df_train_scaled = df_train[feature_columns].copy()
    df_train_scaled['usdkrw(target)'] = usdkrw_scaler.fit_transform(df_train[['usdkrw(target)']])
    
    # 외생변수 정규화
    exogenous_cols = [col for col in feature_columns if col != 'usdkrw(target)']
    df_train_scaled[exogenous_cols] = exogenous_scaler.fit_transform(df_train[exogenous_cols])
    
    # target 데이터는 usdkrw는 원본 값 유지, 외생변수만 정규화
    df_target_scaled = df_target[feature_columns].copy()
    # usdkrw는 원본 값 유지 (정규화하지 않음)
    df_target_scaled[exogenous_cols] = exogenous_scaler.transform(df_target[exogenous_cols])
    
    return df_train_scaled, df_target_scaled, usdkrw_scaler, exogenous_scaler, feature_columns

def create_sequences(data, seq_length=30, pred_length=7):
    """시계열 시퀀스 생성"""
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data) - pred_length + 1):
        # 입력 시퀀스
        seq = data.iloc[i-seq_length:i].values
        # 타겟 시퀀스 (미래 7일)
        target = data.iloc[i:i+pred_length, 0].values  # usdkrw만
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# ============================================================================
# 3. 모델 학습
# ============================================================================

def train_model(model, train_sequences, train_targets, epochs=100, learning_rate=0.001):
    """모델 학습"""
    print("🚀 모델 학습 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    model = model.to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 데이터를 텐서로 변환
    train_sequences_tensor = torch.FloatTensor(train_sequences).to(device)
    train_targets_tensor = torch.FloatTensor(train_targets).to(device)
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 입력 데이터 분리
        endogenous_x = train_sequences_tensor[:, :, 0:1]  # usdkrw만
        exogenous_x = train_sequences_tensor[:, :, 1:]    # 외생변수들
        
        # 예측
        outputs = model(endogenous_x, exogenous_x)
        
        # 손실 계산
        loss = criterion(outputs, train_targets_tensor)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    print("✅ 모델 학습 완료!")
    return model

# ============================================================================
# 4. 예측 및 결과 분석
# ============================================================================

def predict_future(model, last_sequence, usdkrw_scaler, feature_columns, days_to_predict=31, pred_length=7):
    """미래 예측"""
    print(f"🔮 {days_to_predict}일 미래 예측 중...")
    
    device = next(model.parameters()).device
    model.eval()
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    with torch.no_grad():
        for i in range(0, days_to_predict, pred_length):
            # 현재 시퀀스를 텐서로 변환
            seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            
            # 입력 데이터 분리
            endogenous_x = seq_tensor[:, :, 0:1]
            exogenous_x = seq_tensor[:, :, 1:]
            
            # 예측
            pred = model(endogenous_x, exogenous_x)
            pred = pred.squeeze(0).cpu().numpy()
            
            # 예측값을 시퀀스에 추가
            for j, p in enumerate(pred):
                if i + j < days_to_predict:
                    new_row = current_sequence[-1].copy()
                    new_row[0] = p  # usdkrw 예측값
                    current_sequence = np.vstack([current_sequence, new_row])
                    predictions.append(p)
            
            # 시퀀스 길이 유지
            if len(current_sequence) > 30:
                current_sequence = current_sequence[-30:]
    
    # 예측값 역정규화 (usdkrw만)
    predictions_rescaled = []
    for pred in predictions:
        # usdkrw 예측값만 역정규화
        pred_rescaled = usdkrw_scaler.inverse_transform([[pred]])[0, 0]
        predictions_rescaled.append(pred_rescaled)
    
    return predictions_rescaled

def plot_results(actual, predicted, dates):
    """결과 시각화"""
    plt.figure(figsize=(15, 8))
    
    # 실제값과 예측값
    plt.plot(dates, actual, 'b-', label='Actual Values', linewidth=2)
    plt.plot(dates, predicted, 'r--', label='Predicted Values', linewidth=2)
    
    plt.title('LSTM-based Model with TimeXer Ideas: USD/KRW Exchange Rate Prediction (January 2025)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('USD/KRW Exchange Rate', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('lstm_timexer_style_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(actual, predicted, dates):
    """결과 저장"""
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predicted,
        'error': np.array(actual) - np.array(predicted),
        'error_pct': ((np.array(actual) - np.array(predicted)) / np.array(actual)) * 100
    })
    
    results_df.to_csv('lstm_timexer_style_prediction_results.csv', index=False)
    
    # 성능 지표 계산
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 예측 성능:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 요약 통계 저장
    summary = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'prediction_days': len(predicted),
        'model': 'Simple TimeXer'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('lstm_timexer_style_summary.csv', index=False)
    
    print(f"\n💾 Results saved:")
    print(f"- Prediction results: lstm_timexer_style_prediction_results.csv")
    print(f"- Summary statistics: lstm_timexer_style_summary.csv")
    print(f"- Visualization: lstm_timexer_style_prediction_results.png")

# ============================================================================
# 5. 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🚀 LSTM-based Model with TimeXer Ideas: Training on df_trial.csv and Predicting target_trial.csv")
    print("=" * 70)
    
    # 1. 데이터 로드 및 전처리
    df_train_scaled, df_target_scaled, usdkrw_scaler, exogenous_scaler, feature_columns = load_and_preprocess_data()
    
    # 2. 시퀀스 생성
    seq_length = 30
    pred_length = 7
    train_sequences, train_targets = create_sequences(df_train_scaled, seq_length, pred_length)
    print(f"생성된 학습 시퀀스: {train_sequences.shape}")
    print(f"생성된 타겟: {train_targets.shape}")
    
    # 3. 모델 생성 및 학습
    model = SimpleTimeXerModel(
        endogenous_dim=1,
        exogenous_dim=5,
        hidden_size=128,
        num_layers=2,
        output_size=pred_length
    )
    
    model = train_model(model, train_sequences, train_targets, epochs=100, learning_rate=0.001)
    
    # 4. 미래 예측
    last_sequence = df_train_scaled.iloc[-seq_length:].values
    predictions = predict_future(model, last_sequence, usdkrw_scaler, feature_columns, days_to_predict=31, pred_length=pred_length)
    
    # 5. 결과 분석
    actual_values = df_target_scaled['usdkrw(target)'].values
    actual_dates = pd.date_range('2025-01-01', '2025-01-31', freq='D')
    
    print(f"\n📈 예측 결과 요약:")
    print(f"실제값 범위: {actual_values.min():.1f} ~ {actual_values.max():.1f}")
    print(f"예측값 범위: {min(predictions):.1f} ~ {max(predictions):.1f}")
    
    # 6. 결과 시각화 및 저장
    plot_results(actual_values, predictions, actual_dates)
    save_results(actual_values, predictions, actual_dates)
    
    print("\n" + "=" * 70)
    print("🎉 LSTM-based Model with TimeXer Ideas Prediction Completed!")
    print("=" * 70)

if __name__ == "__main__":
    main() 
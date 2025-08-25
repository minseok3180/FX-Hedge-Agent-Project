#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
환 리스크 헷지를 위한 강화학습 환경
Gymnasium 기반의 커스텀 환경으로 구현
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class HedgeAction(Enum):
    """헷지 전략 액션 정의"""
    HOLD = 0              # A0: 유지
    FORWARD_1M_ADD = 1    # A1: 선도 1M로 +25%p 증액
    FORWARD_3M_ADD = 2    # A2: 선도 3M로 +25%p 증액
    FORWARD_REDUCE = 3    # A3: 선도 최근만기에서 -25%p 감액/청산
    FUTURES_ADD = 4       # A4: 선물 근월물로 +25%p 증액
    FUTURES_REDUCE = 5    # A5: 선물 근월물에서 -25%p 감액/청산
    ROLLOVER = 6          # A6: 롤오버
    FLATTEN = 7           # A7: 전량 청산
    SWITCH = 8            # A8: 스위치 25%p

@dataclass
class Position:
    """포지션 정보"""
    forward_1m: float = 0.0      # 1M 선도 포지션
    forward_3m: float = 0.0      # 3M 선도 포지션
    futures: float = 0.0         # 선물 포지션
    hedge_ratio: float = 0.0     # 전체 헷지 비율

class HedgeEnvironment(gym.Env):
    """환 리스크 헷지 강화학습 환경"""
    
    def __init__(self, 
                 data_path: str,
                 initial_capital: float = 1000000.0,
                 max_hedge_ratio: float = 1.0,
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Args:
            data_path: 데이터 파일 경로
            initial_capital: 초기 자본금
            max_hedge_ratio: 최대 헷지 비율
            transaction_cost: 거래 비용 (0.1%)
            risk_free_rate: 무위험 수익률
        """
        super().__init__()
        
        # 데이터 로드 및 전처리
        self.data = self._load_and_preprocess_data(data_path)
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # 환경 파라미터
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_hedge_ratio = max_hedge_ratio
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # 포지션 관리
        self.position = Position()
        self.position_history = []
        self.capital_history = []
        
        # 액션 및 관찰 공간 정의
        self.action_space = gym.spaces.Discrete(9)  # 9가지 헷지 전략
        
        # 상태 공간: [과거 데이터(730일) + 예측 데이터(31일) + 현재 포지션 정보]
        state_dim = 730 + 31 + 4  # 765차원
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 메타데이터
        self.metadata = {'render_modes': ['human', 'rgb_array']}
        
    def _load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        try:
            # yonju 폴더의 데이터와 chaewon 폴더의 예측 결과를 통합
            df = pd.read_csv(data_path)
            
            # 날짜 컬럼 처리
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 결측값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 수치형 컬럼 변환
            numeric_columns = ['usdkrw(target)', 'us_ex', 'us_im', 'reserve', 'us_reserve', 
                             'us_export', 'us_import', 'base', 'market', 'consumer', 
                             'exp_rate', 'im_rate', 'us_gdp', 'us_stock']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 정규화 (Z-score)
            for col in numeric_columns:
                if col in df.columns and col != 'usdkrw(target)':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[col] = (df[col] - mean_val) / std_val
            
            print(f"✅ 데이터 로드 완료: {len(df)}행, {len(df.columns)}컬럼")
            return df
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            # 샘플 데이터 생성 (실제 구현시에는 실제 데이터 사용)
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """샘플 데이터 생성 (테스트용)"""
        dates = pd.date_range('2023-01-01', '2025-01-31', freq='D')
        n_days = len(dates)
        
        # USD/KRW 환율 (실제와 유사한 패턴)
        np.random.seed(42)
        usdkrw_base = 1300
        usdkrw_trend = np.cumsum(np.random.normal(0, 0.5, n_days))
        usdkrw = usdkrw_base + usdkrw_trend + np.random.normal(0, 10, n_days)
        
        # 기타 경제 지표들
        data = {
            'date': dates,
            'usdkrw(target)': usdkrw,
            'us_ex': np.random.normal(0, 1, n_days),
            'us_im': np.random.normal(0, 1, n_days),
            'reserve': np.random.normal(0, 1, n_days),
            'us_reserve': np.random.normal(0, 1, n_days),
            'us_export': np.random.normal(0, 1, n_days),
            'us_import': np.random.normal(0, 1, n_days),
            'base': np.random.normal(0, 1, n_days),
            'market': np.random.normal(0, 1, n_days),
            'consumer': np.random.normal(0, 1, n_days),
            'exp_rate': np.random.normal(0, 1, n_days),
            'im_rate': np.random.normal(0, 1, n_days),
            'us_gdp': np.random.normal(0, 1, n_days),
            'us_stock': np.random.normal(0, 1, n_days)
        }
        
        df = pd.DataFrame(data)
        print("📊 샘플 데이터 생성 완료")
        return df
    
    def _get_state(self) -> np.ndarray:
        """현재 상태 반환"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        # 과거 2년 데이터 (730일)
        start_idx = max(0, self.current_step - 730)
        end_idx = self.current_step + 1
        
        # 과거 데이터 (usdkrw(target)만 사용)
        past_data = self.data.iloc[start_idx:end_idx]['usdkrw(target)'].values
        if len(past_data) < 730:
            past_data = np.pad(past_data, (730 - len(past_data), 0), 'constant')
        else:
            past_data = past_data[-730:]
        
        # 향후 1달 예측 데이터 (31일) - 실제로는 chaewon 모델의 예측값 사용
        future_data = np.zeros(31)
        if self.current_step + 31 < len(self.data):
            future_data = self.data.iloc[self.current_step:self.current_step+31]['usdkrw(target)'].values
        
        # 현재 포지션 정보
        position_info = np.array([
            self.position.forward_1m,
            self.position.forward_3m,
            self.position.futures,
            self.position.hedge_ratio
        ])
        
        # 상태 결합
        state = np.concatenate([past_data, future_data, position_info])
        return state.astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """보상 계산"""
        if self.current_step >= len(self.data) - 1:
            return 0.0
        
        current_usdkrw = self.data.iloc[self.current_step]['usdkrw(target)']
        next_usdkrw = self.data.iloc[self.current_step + 1]['usdkrw(target)']
        
        # 환율 변동률
        exchange_rate_change = (next_usdkrw - current_usdkrw) / current_usdkrw
        
        # 포지션에 따른 수익/손실
        position_pnl = 0.0
        
        # 선도 포지션 P&L
        if self.position.forward_1m != 0:
            position_pnl += self.position.forward_1m * exchange_rate_change
        if self.position.forward_3m != 0:
            position_pnl += self.position.forward_3m * exchange_rate_change
        if self.position.futures != 0:
            position_pnl += self.position.futures * exchange_rate_change
        
        # 거래 비용
        transaction_cost = 0.0
        if action != HedgeAction.HOLD.value:
            transaction_cost = abs(self.transaction_cost * self.current_capital)
        
        # 보상 = 포지션 P&L - 거래 비용
        reward = position_pnl - transaction_cost
        
        # 리스크 조정 보상 (Sharpe ratio 스타일)
        if self.current_step > 0:
            volatility = np.std(self.capital_history[-30:]) if len(self.capital_history) >= 30 else 1.0
            if volatility > 0:
                reward = reward / volatility
        
        return reward
    
    def _execute_action(self, action: int) -> None:
        """액션 실행"""
        if action == HedgeAction.HOLD.value:
            pass  # 아무것도 하지 않음
        
        elif action == HedgeAction.FORWARD_1M_ADD.value:
            # 선도 1M로 +25%p 증액
            add_amount = 0.25 * self.current_capital
            self.position.forward_1m += add_amount
            self.position.hedge_ratio = min(self.max_hedge_ratio, 
                                          (abs(self.position.forward_1m) + abs(self.position.forward_3m) + abs(self.position.futures)) / self.current_capital)
        
        elif action == HedgeAction.FORWARD_3M_ADD.value:
            # 선도 3M로 +25%p 증액
            add_amount = 0.25 * self.current_capital
            self.position.forward_3m += add_amount
            self.position.hedge_ratio = min(self.max_hedge_ratio, 
                                          (abs(self.position.forward_1m) + abs(self.position.forward_3m) + abs(self.position.futures)) / self.current_capital)
        
        elif action == HedgeAction.FORWARD_REDUCE.value:
            # 선도 최근만기에서 -25%p 감액/청산
            reduce_amount = 0.25 * self.current_capital
            if self.position.forward_1m > 0:
                self.position.forward_1m = max(0, self.position.forward_1m - reduce_amount)
            elif self.position.forward_3m > 0:
                self.position.forward_3m = max(0, self.position.forward_3m - reduce_amount)
            
            self.position.hedge_ratio = (abs(self.position.forward_1m) + abs(self.position.forward_3m) + abs(self.position.futures)) / self.current_capital
        
        elif action == HedgeAction.FUTURES_ADD.value:
            # 선물 근월물로 +25%p 증액
            add_amount = 0.25 * self.current_capital
            self.position.futures += add_amount
            self.position.hedge_ratio = min(self.max_hedge_ratio, 
                                          (abs(self.position.forward_1m) + abs(self.position.forward_3m) + abs(self.position.futures)) / self.current_capital)
        
        elif action == HedgeAction.FUTURES_REDUCE.value:
            # 선물 근월물에서 -25%p 감액/청산
            reduce_amount = 0.25 * self.current_capital
            self.position.futures = max(0, self.position.futures - reduce_amount)
            self.position.hedge_ratio = (abs(self.position.forward_1m) + abs(self.position.forward_3m) + abs(self.position.futures)) / self.current_capital
        
        elif action == HedgeAction.ROLLOVER.value:
            # 롤오버: 최근 만기 포지션을 다음 만기로 전부 이월
            if self.position.forward_1m > 0:
                self.position.forward_3m += self.position.forward_1m
                self.position.forward_1m = 0
        
        elif action == HedgeAction.FLATTEN.value:
            # 전량 청산
            self.position.forward_1m = 0
            self.position.forward_3m = 0
            self.position.futures = 0
            self.position.hedge_ratio = 0
        
        elif action == HedgeAction.SWITCH.value:
            # 스위치 25%p: 선도 ↔ 선물로 일부 전환
            switch_amount = 0.25 * self.current_capital
            if self.position.forward_1m > switch_amount:
                self.position.forward_1m -= switch_amount
                self.position.futures += switch_amount
            elif self.position.futures > switch_amount:
                self.position.futures -= switch_amount
                self.position.forward_1m += switch_amount
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """환경 스텝 실행"""
        # 액션 실행
        self._execute_action(action)
        
        # 보상 계산
        reward = self._calculate_reward(action)
        
        # 자본금 업데이트
        self.current_capital += reward
        
        # 포지션 및 자본금 히스토리 저장
        self.position_history.append(Position(
            forward_1m=self.position.forward_1m,
            forward_3m=self.position.forward_3m,
            futures=self.position.futures,
            hedge_ratio=self.position.hedge_ratio
        ))
        self.capital_history.append(self.current_capital)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 종료 조건 확인
        done = self.current_step >= self.max_steps
        truncated = False
        
        # 다음 상태
        next_state = self._get_state()
        
        # 정보
        info = {
            'current_capital': self.current_capital,
            'hedge_ratio': self.position.hedge_ratio,
            'position': {
                'forward_1m': self.position.forward_1m,
                'forward_3m': self.position.forward_3m,
                'futures': self.position.futures
            },
            'step': self.current_step,
            'reward': reward
        }
        
        return next_state, reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 상태 초기화
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.position = Position()
        self.position_history = []
        self.capital_history = [self.initial_capital]
        
        # 초기 상태 반환
        initial_state = self._get_state()
        info = {
            'current_capital': self.current_capital,
            'hedge_ratio': self.position.hedge_ratio,
            'position': {
                'forward_1m': self.position.forward_1m,
                'forward_3m': self.position.forward_3m,
                'futures': self.position.futures
            },
            'step': self.current_step
        }
        
        return initial_state, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """환경 렌더링"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Capital: ${self.current_capital:,.2f}")
            print(f"Hedge Ratio: {self.position.hedge_ratio:.2%}")
            print(f"Positions: 1M Forward: {self.position.forward_1m:,.2f}, "
                  f"3M Forward: {self.position.forward_3m:,.2f}, "
                  f"Futures: {self.position.futures:,.2f}")
            return None
        elif mode == 'rgb_array':
            # 시각화 이미지 반환 (구현 필요)
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self) -> None:
        """환경 종료"""
        pass

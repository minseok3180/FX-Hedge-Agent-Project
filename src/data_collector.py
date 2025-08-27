import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Any

class FXDataCollector:
    """외환 데이터 수집 및 처리 클래스 (KRW 기반)"""
    
    def __init__(self):
        # KRW 기반 주요 통화쌍
        self.major_pairs = [
            'USDKRW=X', 'EURKRW=X', 'JPYKRW=X', 'GBPKRW=X',
            'CNYKRW=X', 'AUDKRW=X', 'CADKRW=X', 'CHFKRW=X'
        ]
        
        # 통화쌍별 한국어 이름
        self.pair_names = {
            'USDKRW=X': '달러/원',
            'EURKRW=X': '유로/원',
            'JPYKRW=X': '엔/원',
            'GBPKRW=X': '파운드/원',
            'CNYKRW=X': '위안/원',
            'AUDKRW=X': '호주달러/원',
            'CADKRW=X': '캐나다달러/원',
            'CHFKRW=X': '스위스프랑/원'
        }
        
    def get_fx_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Yahoo Finance에서 외환 데이터 수집"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"데이터 수집 오류 ({symbol}): {e}")
            return pd.DataFrame()
    
    def get_all_major_pairs(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """모든 주요 통화쌍 데이터 수집"""
        data_dict = {}
        for pair in self.major_pairs:
            data = self.get_fx_data(pair, period)
            if not data.empty:
                data_dict[pair] = data
        return data_dict
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        if df.empty:
            return df
            
        # 이동평균
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        df['BB_upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        return df
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, float]:
        """시장 심리 지표 계산"""
        data = self.get_fx_data(symbol, "6mo")
        if data.empty:
            return {}
            
        data = self.calculate_technical_indicators(data)
        
        # 최신 데이터
        latest = data.iloc[-1]
        
        sentiment = {
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd_signal': 1 if latest['MACD'] > latest['MACD_signal'] else -1,
            'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
            'trend': 1 if latest['MA_20'] > latest['MA_50'] else -1
        }
        
        return sentiment
    
    def create_market_summary(self) -> str:
        """시장 요약 정보 생성"""
        summary_parts = []
        
        for pair in self.major_pairs:
            sentiment = self.get_market_sentiment(pair)
            if sentiment:
                pair_name = self.pair_names.get(pair, pair.replace('=X', ''))
                summary_parts.append(
                    f"{pair_name}: 가격 {sentiment['price']:.2f}, "
                    f"RSI {sentiment['rsi']:.1f}, "
                    f"MACD 신호 {'매수' if sentiment['macd_signal'] > 0 else '매도'}, "
                    f"볼린저밴드 위치 {sentiment['bb_position']:.2f}, "
                    f"추세 {'상승' if sentiment['trend'] > 0 else '하락'}"
                )
        
        return "\n".join(summary_parts)
    
    def get_krw_exchange_rate(self, currency: str = "USD") -> float:
        """특정 통화의 KRW 환율 조회"""
        if currency == "KRW":
            return 1.0
            
        symbol = f"{currency}KRW=X"
        data = self.get_fx_data(symbol, "1d")
        
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            return 0.0
    
    def get_korean_market_info(self) -> Dict[str, Any]:
        """한국 시장 정보 조회"""
        market_info = {}
        
        # 주요 통화별 환율
        for pair in self.major_pairs:
            data = self.get_fx_data(pair, "1d")
            if not data.empty:
                currency = pair.replace('KRW=X', '')
                market_info[currency] = {
                    'current_rate': data['Close'].iloc[-1],
                    'change': data['Close'].iloc[-1] - data['Open'].iloc[0],
                    'change_percent': ((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
                }
        
        return market_info

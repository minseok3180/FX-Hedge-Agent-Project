import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

class FXAgent:
    """외환 거래 지능형 에이전트"""
    
    def __init__(self, rag_system, data_collector):
        self.rag_system = rag_system
        self.data_collector = data_collector
        self.trading_history = []
        self.current_positions = {}
        
    def analyze_market_condition(self, symbol: str) -> Dict[str, Any]:
        """시장 상황 분석"""
        # 시장 데이터 수집
        data = self.data_collector.get_fx_data(symbol, "3mo")
        if data.empty:
            return {"error": "데이터를 가져올 수 없습니다."}
        
        # 기술적 지표 계산
        data = self.data_collector.calculate_technical_indicators(data)
        
        # 최신 데이터
        latest = data.iloc[-1]
        
        # 시장 상황 분석
        analysis = {
            "symbol": symbol,
            "current_price": latest['Close'],
            "trend": "상승" if latest['MA_20'] > latest['MA_50'] else "하락",
            "rsi_signal": "과매수" if latest['RSI'] > 70 else "과매도" if latest['RSI'] < 30 else "중립",
            "macd_signal": "매수" if latest['MACD'] > latest['MACD_signal'] else "매도",
            "bb_position": "상단" if latest['Close'] > latest['BB_upper'] else "하단" if latest['Close'] < latest['BB_lower'] else "중간",
            "volatility": data['Close'].pct_change().std() * np.sqrt(252) * 100,  # 연간 변동성
            "support_level": latest['BB_lower'],
            "resistance_level": latest['BB_upper']
        }
        
        return analysis
    
    def generate_trading_recommendation(self, symbol: str) -> Dict[str, Any]:
        """거래 추천 생성"""
        # 시장 분석
        analysis = self.analyze_market_condition(symbol)
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        # RAG 시스템을 통한 전문가 조언 요청
        query = f"{symbol} 통화쌍의 현재 시장 상황을 분석하고 거래 전략을 제시해주세요. 현재 가격: {analysis['current_price']}, RSI: {analysis['rsi_signal']}, MACD: {analysis['macd_signal']}"
        
        try:
            expert_advice = self.rag_system.rag_query(query)
        except Exception as e:
            expert_advice = "전문가 조언을 가져올 수 없습니다."
        
        # 거래 신호 생성
        signal_strength = 0
        signal = "관망"
        
        # RSI 기반 신호
        if analysis['rsi_signal'] == "과매도":
            signal_strength += 1
        elif analysis['rsi_signal'] == "과매수":
            signal_strength -= 1
        
        # MACD 기반 신호
        if analysis['macd_signal'] == "매수":
            signal_strength += 1
        elif analysis['macd_signal'] == "매도":
            signal_strength -= 1
        
        # 추세 기반 신호
        if analysis['trend'] == "상승":
            signal_strength += 0.5
        else:
            signal_strength -= 0.5
        
        # 최종 신호 결정
        if signal_strength >= 1.5:
            signal = "강력 매수"
        elif signal_strength >= 0.5:
            signal = "매수"
        elif signal_strength <= -1.5:
            signal = "강력 매도"
        elif signal_strength <= -0.5:
            signal = "매도"
        
        recommendation = {
            "symbol": symbol,
            "signal": signal,
            "signal_strength": signal_strength,
            "current_price": analysis['current_price'],
            "target_price": self._calculate_target_price(analysis, signal),
            "stop_loss": self._calculate_stop_loss(analysis, signal),
            "confidence": min(abs(signal_strength) * 25, 100),  # 신뢰도 (0-100%)
            "reasoning": f"RSI: {analysis['rsi_signal']}, MACD: {analysis['macd_signal']}, 추세: {analysis['trend']}",
            "expert_advice": expert_advice,
            "timestamp": datetime.now().isoformat()
        }
        
        return recommendation
    
    def _calculate_target_price(self, analysis: Dict, signal: str) -> float:
        """목표 가격 계산"""
        current_price = analysis['current_price']
        volatility = analysis['volatility'] / 100
        
        if "매수" in signal:
            # 상승 목표: 현재가 + 변동성의 1.5배
            return current_price * (1 + volatility * 1.5)
        elif "매도" in signal:
            # 하락 목표: 현재가 - 변동성의 1.5배
            return current_price * (1 - volatility * 1.5)
        else:
            return current_price
    
    def _calculate_stop_loss(self, analysis: Dict, signal: str) -> float:
        """손절매 가격 계산"""
        current_price = analysis['current_price']
        volatility = analysis['volatility'] / 100
        
        if "매수" in signal:
            # 매수 시 손절매: 현재가 - 변동성의 1배
            return current_price * (1 - volatility)
        elif "매도" in signal:
            # 매도 시 손절매: 현재가 + 변동성의 1배
            return current_price * (1 + volatility)
        else:
            return current_price
    
    def execute_trade(self, recommendation: Dict, position_size: float = 1.0) -> Dict[str, Any]:
        """거래 실행 (시뮬레이션)"""
        if "error" in recommendation:
            return {"error": recommendation["error"]}
        
        trade = {
            "id": len(self.trading_history) + 1,
            "symbol": recommendation["symbol"],
            "signal": recommendation["signal"],
            "entry_price": recommendation["current_price"],
            "target_price": recommendation["target_price"],
            "stop_loss": recommendation["stop_loss"],
            "position_size": position_size,
            "entry_time": datetime.now(),
            "status": "진행중",
            "pnl": 0.0,
            "pnl_percentage": 0.0
        }
        
        # 포지션 기록
        self.current_positions[recommendation["symbol"]] = trade
        self.trading_history.append(trade)
        
        return {
            "message": f"{recommendation['symbol']} {recommendation['signal']} 거래가 실행되었습니다.",
            "trade_details": trade
        }
    
    def update_positions(self) -> List[Dict]:
        """포지션 상태 업데이트"""
        updated_positions = []
        
        for symbol, position in self.current_positions.items():
            if position["status"] != "진행중":
                continue
                
            # 현재 가격 확인
            current_data = self.data_collector.get_fx_data(symbol, "1d")
            if current_data.empty:
                continue
                
            current_price = current_data.iloc[-1]['Close']
            
            # 손익 계산
            if "매수" in position["signal"]:
                pnl = (current_price - position["entry_price"]) * position["position_size"]
                pnl_percentage = (pnl / (position["entry_price"] * position["position_size"])) * 100
            else:  # 매도
                pnl = (position["entry_price"] - current_price) * position["position_size"]
                pnl_percentage = (pnl / (position["entry_price"] * position["position_size"])) * 100
            
            # 포지션 업데이트
            position["current_price"] = current_price
            position["pnl"] = pnl
            position["pnl_percentage"] = pnl_percentage
            
            # 청산 조건 확인
            if "매수" in position["signal"]:
                if current_price >= position["target_price"]:
                    position["status"] = "목표 달성"
                elif current_price <= position["stop_loss"]:
                    position["status"] = "손절매"
            else:  # 매도
                if current_price <= position["target_price"]:
                    position["status"] = "목표 달성"
                elif current_price >= position["stop_loss"]:
                    position["status"] = "손절매"
            
            updated_positions.append(position)
        
        return updated_positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """포트폴리오 요약"""
        active_positions = [p for p in self.current_positions.values() if p["status"] == "진행중"]
        closed_positions = [p for p in self.trading_history if p["status"] != "진행중"]
        
        total_pnl = sum(p["pnl"] for p in closed_positions)
        total_pnl_percentage = sum(p["pnl_percentage"] for p in closed_positions)
        
        if closed_positions:
            avg_pnl_percentage = total_pnl_percentage / len(closed_positions)
        else:
            avg_pnl_percentage = 0
        
        return {
            "active_positions": len(active_positions),
            "closed_positions": len(closed_positions),
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_percentage,
            "average_pnl_percentage": avg_pnl_percentage,
            "win_rate": len([p for p in closed_positions if p["pnl"] > 0]) / len(closed_positions) if closed_positions else 0
        }

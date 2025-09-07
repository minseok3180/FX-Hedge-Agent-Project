"""
CSV 데이터 관리 모듈
RDB 대신 CSV 파일을 사용하여 데이터를 관리
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from config import Config


class CSVDataManager:
    """CSV 데이터 관리자"""
    
    def __init__(self, config: Config):
        self.config = config.db
        self.logger = logging.getLogger(__name__)
        
        # CSV 디렉토리 생성
        self._ensure_data_directory()
        
        # CSV 파일 경로들
        self.users_file = os.path.join(self.config.csv_data_dir, self.config.user_data_file)
        self.trading_history_file = os.path.join(self.config.csv_data_dir, self.config.trading_history_file)
        self.portfolio_file = os.path.join(self.config.csv_data_dir, self.config.portfolio_file)
        
        # TODO: 초기 CSV 파일 생성
        self._initialize_csv_files()
    
    def _ensure_data_directory(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.config.csv_data_dir):
            os.makedirs(self.config.csv_data_dir, exist_ok=True)
            self.logger.info(f"데이터 디렉토리 생성: {self.config.csv_data_dir}")
    
    def _initialize_csv_files(self):
        """초기 CSV 파일들 생성"""
        try:
            # 사용자 데이터 파일
            if not os.path.exists(self.users_file):
                users_df = pd.DataFrame(columns=[
                    'user_id', 'username', 'email', 'created_at', 'last_login',
                    'risk_tolerance', 'preferences', 'status'
                ])
                users_df.to_csv(self.users_file, index=False)
                self.logger.info(f"사용자 데이터 파일 생성: {self.users_file}")
            
            # 거래 히스토리 파일
            if not os.path.exists(self.trading_history_file):
                trading_df = pd.DataFrame(columns=[
                    'trade_id', 'user_id', 'currency_pair', 'order_type',
                    'amount', 'price', 'strategy_id', 'status', 'created_at',
                    'executed_at', 'profit_loss', 'fees'
                ])
                trading_df.to_csv(self.trading_history_file, index=False)
                self.logger.info(f"거래 히스토리 파일 생성: {self.trading_history_file}")
            
            # 포트폴리오 파일
            if not os.path.exists(self.portfolio_file):
                portfolio_df = pd.DataFrame(columns=[
                    'user_id', 'currency', 'amount', 'avg_price',
                    'last_updated', 'portfolio_value'
                ])
                portfolio_df.to_csv(self.portfolio_file, index=False)
                self.logger.info(f"포트폴리오 파일 생성: {self.portfolio_file}")
                
        except Exception as e:
            self.logger.error(f"CSV 파일 초기화 실패: {str(e)}")
            raise e
    
    # 사용자 관리
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """
        새 사용자 생성
        
        Args:
            user_data: 사용자 데이터
            
        Returns:
            생성 성공 여부
        """
        try:
            df = pd.read_csv(self.users_file)
            
            # 사용자 ID 중복 확인
            if 'user_id' in user_data and user_data['user_id'] in df['user_id'].values:
                self.logger.warning(f"사용자 ID가 이미 존재합니다: {user_data['user_id']}")
                return False
            
            # 새 사용자 데이터 추가
            new_user = {
                'user_id': user_data.get('user_id', f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'username': user_data.get('username', ''),
                'email': user_data.get('email', ''),
                'created_at': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat(),
                'risk_tolerance': user_data.get('risk_tolerance', 'medium'),
                'preferences': json.dumps(user_data.get('preferences', {})),
                'status': 'active'
            }
            
            df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
            df.to_csv(self.users_file, index=False)
            
            self.logger.info(f"새 사용자 생성: {new_user['user_id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 생성 실패: {str(e)}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        사용자 정보 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            사용자 정보 또는 None
        """
        try:
            df = pd.read_csv(self.users_file)
            user_row = df[df['user_id'] == user_id]
            
            if user_row.empty:
                return None
            
            user_data = user_row.iloc[0].to_dict()
            
            # JSON 필드 파싱
            if 'preferences' in user_data and user_data['preferences']:
                try:
                    user_data['preferences'] = json.loads(user_data['preferences'])
                except json.JSONDecodeError:
                    user_data['preferences'] = {}
            
            return user_data
            
        except Exception as e:
            self.logger.error(f"사용자 조회 실패: {str(e)}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """
        사용자 정보 업데이트
        
        Args:
            user_id: 사용자 ID
            update_data: 업데이트할 데이터
            
        Returns:
            업데이트 성공 여부
        """
        try:
            df = pd.read_csv(self.users_file)
            
            if user_id not in df['user_id'].values:
                self.logger.warning(f"사용자를 찾을 수 없습니다: {user_id}")
                return False
            
            # 업데이트할 필드들 처리
            for key, value in update_data.items():
                if key == 'preferences' and isinstance(value, dict):
                    value = json.dumps(value)
                df.loc[df['user_id'] == user_id, key] = value
            
            df.to_csv(self.users_file, index=False)
            self.logger.info(f"사용자 정보 업데이트: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 업데이트 실패: {str(e)}")
            return False
    
    # 포트폴리오 관리
    def get_user_portfolio(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 포트폴리오 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            포트폴리오 정보
        """
        try:
            df = pd.read_csv(self.portfolio_file)
            user_portfolio = df[df['user_id'] == user_id]
            
            if user_portfolio.empty:
                return {
                    "user_id": user_id,
                    "total_assets": {"KRW": 0.0, "USD": 0.0, "EUR": 0.0, "JPY": 0.0},
                    "portfolio_composition": {},
                    "risk_exposure": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
            
            # 통화별 자산 계산
            total_assets = {}
            portfolio_composition = {}
            
            for _, row in user_portfolio.iterrows():
                currency = row['currency']
                amount = float(row['amount'])
                total_assets[currency] = amount
                portfolio_composition[currency] = {
                    'amount': amount,
                    'avg_price': float(row['avg_price']),
                    'value': amount * float(row['avg_price'])
                }
            
            return {
                "user_id": user_id,
                "total_assets": total_assets,
                "portfolio_composition": portfolio_composition,
                "risk_exposure": 0.0,  # TODO: 리스크 노출도 계산
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 조회 실패: {str(e)}")
            return {
                "user_id": user_id,
                "total_assets": {},
                "portfolio_composition": {},
                "risk_exposure": 0.0,
                "last_updated": datetime.now().isoformat()
            }
    
    def update_portfolio(self, user_id: str, currency: str, amount: float, avg_price: float) -> bool:
        """
        포트폴리오 업데이트
        
        Args:
            user_id: 사용자 ID
            currency: 통화
            amount: 수량
            avg_price: 평균 가격
            
        Returns:
            업데이트 성공 여부
        """
        try:
            df = pd.read_csv(self.portfolio_file)
            
            # 기존 포지션 확인
            existing_row = df[(df['user_id'] == user_id) & (df['currency'] == currency)]
            
            if not existing_row.empty:
                # 기존 포지션 업데이트
                idx = existing_row.index[0]
                old_amount = float(df.loc[idx, 'amount'])
                old_avg_price = float(df.loc[idx, 'avg_price'])
                
                # 새로운 평균 가격 계산
                new_amount = old_amount + amount
                if new_amount != 0:
                    new_avg_price = ((old_amount * old_avg_price) + (amount * avg_price)) / new_amount
                else:
                    new_avg_price = 0
                
                df.loc[idx, 'amount'] = new_amount
                df.loc[idx, 'avg_price'] = new_avg_price
                df.loc[idx, 'last_updated'] = datetime.now().isoformat()
                
                # 수량이 0이면 행 삭제
                if new_amount == 0:
                    df = df.drop(idx)
            else:
                # 새 포지션 추가
                if amount != 0:
                    new_row = {
                        'user_id': user_id,
                        'currency': currency,
                        'amount': amount,
                        'avg_price': avg_price,
                        'last_updated': datetime.now().isoformat(),
                        'portfolio_value': amount * avg_price
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            df.to_csv(self.portfolio_file, index=False)
            self.logger.info(f"포트폴리오 업데이트: {user_id} - {currency}")
            return True
            
        except Exception as e:
            self.logger.error(f"포트폴리오 업데이트 실패: {str(e)}")
            return False
    
    # 거래 히스토리 관리
    def add_trading_record(self, trade_data: Dict[str, Any]) -> bool:
        """
        거래 기록 추가
        
        Args:
            trade_data: 거래 데이터
            
        Returns:
            추가 성공 여부
        """
        try:
            df = pd.read_csv(self.trading_history_file)
            
            new_trade = {
                'trade_id': trade_data.get('trade_id', f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'user_id': trade_data.get('user_id', ''),
                'currency_pair': trade_data.get('currency_pair', ''),
                'order_type': trade_data.get('order_type', ''),
                'amount': trade_data.get('amount', 0.0),
                'price': trade_data.get('price', 0.0),
                'strategy_id': trade_data.get('strategy_id', ''),
                'status': trade_data.get('status', 'pending'),
                'created_at': datetime.now().isoformat(),
                'executed_at': trade_data.get('executed_at', ''),
                'profit_loss': trade_data.get('profit_loss', 0.0),
                'fees': trade_data.get('fees', 0.0)
            }
            
            df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
            df.to_csv(self.trading_history_file, index=False)
            
            self.logger.info(f"거래 기록 추가: {new_trade['trade_id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"거래 기록 추가 실패: {str(e)}")
            return False
    
    def get_trading_history(self, user_id: str, start_date: str = None, end_date: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        거래 히스토리 조회
        
        Args:
            user_id: 사용자 ID
            start_date: 시작 날짜
            end_date: 종료 날짜
            limit: 조회 제한 개수
            
        Returns:
            거래 히스토리
        """
        try:
            df = pd.read_csv(self.trading_history_file)
            user_trades = df[df['user_id'] == user_id]
            
            # 날짜 필터링
            if start_date:
                user_trades = user_trades[user_trades['created_at'] >= start_date]
            if end_date:
                user_trades = user_trades[user_trades['created_at'] <= end_date]
            
            # 최신순 정렬 및 제한
            user_trades = user_trades.sort_values('created_at', ascending=False).head(limit)
            
            # 통계 계산
            total_trades = len(user_trades)
            successful_trades = len(user_trades[user_trades['status'] == 'executed'])
            total_volume = user_trades['amount'].sum()
            total_profit_loss = user_trades['profit_loss'].sum()
            
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            average_return = (total_profit_loss / total_trades) if total_trades > 0 else 0
            
            return {
                "user_id": user_id,
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "total_volume": float(total_volume),
                "total_profit_loss": float(total_profit_loss),
                "trades": user_trades.to_dict('records'),
                "performance_metrics": {
                    "win_rate": win_rate,
                    "average_return": average_return,
                    "sharpe_ratio": 0.0  # TODO: 샤프 비율 계산
                }
            }
            
        except Exception as e:
            self.logger.error(f"거래 히스토리 조회 실패: {str(e)}")
            return {
                "user_id": user_id,
                "total_trades": 0,
                "successful_trades": 0,
                "total_volume": 0.0,
                "total_profit_loss": 0.0,
                "trades": [],
                "performance_metrics": {
                    "win_rate": 0.0,
                    "average_return": 0.0,
                    "sharpe_ratio": 0.0
                }
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보 반환"""
        try:
            users_df = pd.read_csv(self.users_file)
            trading_df = pd.read_csv(self.trading_history_file)
            portfolio_df = pd.read_csv(self.portfolio_file)
            
            return {
                "total_users": len(users_df),
                "total_trades": len(trading_df),
                "total_portfolios": len(portfolio_df),
                "active_users": len(users_df[users_df['status'] == 'active']),
                "data_files": {
                    "users": self.users_file,
                    "trading_history": self.trading_history_file,
                    "portfolio": self.portfolio_file
                }
            }
            
        except Exception as e:
            self.logger.error(f"데이터 요약 조회 실패: {str(e)}")
            return {
                "total_users": 0,
                "total_trades": 0,
                "total_portfolios": 0,
                "active_users": 0,
                "data_files": {}
            }

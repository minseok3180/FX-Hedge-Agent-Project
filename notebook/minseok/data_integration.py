#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yonju 폴더의 데이터 수집 결과와 chaewon 폴더의 예측 결과를 통합
강화학습 에이전트에 필요한 통합 데이터셋 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class DataIntegrator:
    """데이터 통합 클래스"""
    
    def __init__(self, yonju_data_path: str = None, chaewon_pred_path: str = None):
        """
        Args:
            yonju_data_path: yonju 폴더의 데이터 파일 경로
            chaewon_pred_path: chaewon 폴더의 예측 결과 파일 경로
        """
        self.yonju_data_path = yonju_data_path or str(project_root / "notebook" / "yonju" / "df.csv")
        self.chaewon_pred_path = chaewon_pred_path or str(project_root / "notebook" / "chaewon" / "target_trial.csv")
        
        # 데이터 저장 경로
        self.output_dir = Path(__file__).parent / "integrated_data"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 데이터 통합 시작")
        print(f"   yonju 데이터: {self.yonju_data_path}")
        print(f"   chaewon 예측: {self.chaewon_pred_path}")
        print(f"   출력 디렉토리: {self.output_dir}")
    
    def load_yonju_data(self) -> pd.DataFrame:
        """yonju 폴더의 경제 데이터 로드"""
        try:
            print("📊 yonju 데이터 로드 중...")
            df = pd.read_csv(self.yonju_data_path)
            
            # 날짜 컬럼 처리
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 컬럼명 정리
            column_mapping = {
                'usdkrw(target)': 'usdkrw(target)',
                'us_ex': 'us_ex',
                'us_im': 'us_im',
                'reserve': 'reserve',
                'us_reserve': 'us_reserve',
                'us_export': 'us_export',
                'us_import': 'us_import',
                'base': 'base',
                'market': 'market',
                'consumer': 'consumer',
                'exp_rate': 'exp_rate',
                'im_rate': 'im_rate',
                'us_gdp': 'us_gdp',
                'us_stock': 'us_stock'
            }
            
            # 필요한 컬럼만 선택하고 이름 변경
            available_columns = [col for col in column_mapping.keys() if col in df.columns]
            df = df[['date'] + available_columns].copy()
            
            # 결측값 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✅ yonju 데이터 로드 완료: {len(df)}행, {len(df.columns)}컬럼")
            return df
            
        except Exception as e:
            print(f"❌ yonju 데이터 로드 실패: {e}")
            return self._generate_sample_yonju_data()
    
    def load_chaewon_predictions(self) -> pd.DataFrame:
        """chaewon 폴더의 예측 결과 로드"""
        try:
            print("🔮 chaewon 예측 결과 로드 중...")
            df = pd.read_csv(self.chaewon_pred_path)
            
            # 날짜 컬럼 처리
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 예측값 컬럼 확인 및 처리
            if 'usdkrw(target)' in df.columns:
                df = df[['date', 'usdkrw(target)']].copy()
                df.columns = ['date', 'predicted_usekrw']
            elif 'predicted_usekrw' in df.columns:
                df = df[['date', 'predicted_usekrw']].copy()
            else:
                # 예측값 컬럼이 없는 경우 생성
                df['predicted_usekrw'] = df.iloc[:, 1] if len(df.columns) > 1 else 1300
            
            print(f"✅ chaewon 예측 결과 로드 완료: {len(df)}행")
            return df
            
        except Exception as e:
            print(f"❌ chaewon 예측 결과 로드 실패: {e}")
            return self._generate_sample_chaewon_data()
    
    def _generate_sample_yonju_data(self) -> pd.DataFrame:
        """샘플 yonju 데이터 생성 (테스트용)"""
        print("📊 샘플 yonju 데이터 생성 중...")
        
        # 2023-01-01부터 2024-12-31까지의 일별 데이터
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        
        # USD/KRW 환율 (실제와 유사한 패턴)
        usdkrw_base = 1300
        usdkrw_trend = np.cumsum(np.random.normal(0, 0.3, n_days))
        usdkrw = usdkrw_base + usdkrw_trend + np.random.normal(0, 8, n_days)
        
        # 기타 경제 지표들 (정규화된 값)
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
        print("✅ 샘플 yonju 데이터 생성 완료")
        return df
    
    def _generate_sample_chaewon_data(self) -> pd.DataFrame:
        """샘플 chaewon 예측 데이터 생성 (테스트용)"""
        print("🔮 샘플 chaewon 예측 데이터 생성 중...")
        
        # 2025-01-01부터 2025-01-31까지의 일별 예측 데이터
        dates = pd.date_range('2025-01-01', '2025-01-31', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        
        # 예측된 USD/KRW 환율
        predicted_base = 1320  # 2024년 말 대비 약간 상승
        predicted_trend = np.cumsum(np.random.normal(0, 0.2, n_days))
        predicted_usekrw = predicted_base + predicted_trend + np.random.normal(0, 5, n_days)
        
        data = {
            'date': dates,
            'predicted_usekrw': predicted_usekrw
        }
        
        df = pd.DataFrame(data)
        print("✅ 샘플 chaewon 예측 데이터 생성 완료")
        return df
    
    def integrate_data(self) -> pd.DataFrame:
        """데이터 통합"""
        print("🔗 데이터 통합 중...")
        
        # 데이터 로드
        yonju_df = self.load_yonju_data()
        chaewon_df = self.load_chaewon_predictions()
        
        # 날짜 범위 확인
        yonju_start = yonju_df['date'].min()
        yonju_end = yonju_df['date'].max()
        chaewon_start = chaewon_df['date'].min()
        chaewon_end = chaewon_df['date'].max()
        
        print(f"📅 yonju 데이터 기간: {yonju_start.date()} ~ {yonju_end.date()}")
        print(f"📅 chaewon 예측 기간: {chaewon_start.date()} ~ {chaewon_end.date()}")
        
        # 데이터 통합
        # 1. yonju 데이터를 기준으로 통합
        integrated_df = yonju_df.copy()
        
        # 2. chaewon 예측 데이터 추가
        integrated_df = pd.merge(integrated_df, chaewon_df, on='date', how='left')
        
        # 3. 예측값이 없는 기간은 실제값으로 채움
        integrated_df['predicted_usekrw'] = integrated_df['predicted_usekrw'].fillna(integrated_df['usdkrw(target)'])
        
        # 4. 날짜순 정렬
        integrated_df = integrated_df.sort_values('date').reset_index(drop=True)
        
        # 5. 결측값 처리
        integrated_df = integrated_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ 데이터 통합 완료: {len(integrated_df)}행, {len(integrated_df.columns)}컬럼")
        
        return integrated_df
    
    def create_training_data(self, integrated_df: pd.DataFrame) -> pd.DataFrame:
        """강화학습용 훈련 데이터 생성"""
        print("🎯 강화학습용 훈련 데이터 생성 중...")
        
        # 필요한 컬럼만 선택
        required_columns = [
            'date', 'usdkrw(target)', 'predicted_usekrw',
            'us_ex', 'us_im', 'reserve', 'us_reserve',
            'us_export', 'us_import', 'base', 'market',
            'consumer', 'exp_rate', 'im_rate', 'us_gdp', 'us_stock'
        ]
        
        # 사용 가능한 컬럼만 선택
        available_columns = [col for col in required_columns if col in integrated_df.columns]
        training_df = integrated_df[available_columns].copy()
        
        # 데이터 품질 검사
        print(f"📊 데이터 품질 검사:")
        print(f"   전체 행 수: {len(training_df)}")
        print(f"   결측값 개수: {training_df.isnull().sum().sum()}")
        print(f"   중복 행 수: {training_df.duplicated().sum()}")
        
        # 결측값이 있는 행 제거
        training_df = training_df.dropna()
        print(f"   결측값 제거 후 행 수: {len(training_df)}")
        
        # 날짜 범위 확인
        date_range = training_df['date'].max() - training_df['date'].min()
        print(f"   데이터 기간: {date_range.days}일")
        
        return training_df
    
    def save_integrated_data(self, df: pd.DataFrame, filename: str = "integrated_hedge_data.csv") -> str:
        """통합된 데이터 저장"""
        output_path = self.output_dir / filename
        
        # CSV로 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 통합 데이터 저장 완료: {output_path}")
        
        # 데이터 요약 정보 저장
        summary_path = self.output_dir / "data_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 환 리스크 헷지 강화학습 데이터 요약 ===\n\n")
            f.write(f"데이터 기간: {df['date'].min().date()} ~ {df['date'].max().date()}\n")
            f.write(f"전체 행 수: {len(df):,}\n")
            f.write(f"컬럼 수: {len(df.columns)}\n")
            f.write(f"컬럼 목록: {', '.join(df.columns)}\n\n")
            
            f.write("=== 수치형 컬럼 통계 ===\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'date':
                    f.write(f"{col}:\n")
                    f.write(f"  평균: {df[col].mean():.4f}\n")
                    f.write(f"  표준편차: {df[col].std():.4f}\n")
                    f.write(f"  최소값: {df[col].min():.4f}\n")
                    f.write(f"  최대값: {df[col].max():.4f}\n\n")
        
        print(f"📋 데이터 요약 저장 완료: {summary_path}")
        
        return str(output_path)
    
    def run_integration(self) -> str:
        """전체 데이터 통합 프로세스 실행"""
        print("🚀 데이터 통합 프로세스 시작\n")
        
        # 1. 데이터 통합
        integrated_df = self.integrate_data()
        
        # 2. 훈련 데이터 생성
        training_df = self.create_training_data(integrated_df)
        
        # 3. 결과 저장
        output_path = self.save_integrated_data(training_df)
        
        print(f"\n🎉 데이터 통합 완료!")
        print(f"   최종 데이터 경로: {output_path}")
        print(f"   데이터 크기: {len(training_df):,}행 × {len(training_df.columns)}컬럼")
        
        return output_path

def main():
    """메인 실행 함수"""
    # 데이터 통합기 생성
    integrator = DataIntegrator()
    
    # 데이터 통합 실행
    output_path = integrator.run_integration()
    
    print(f"\n✅ 모든 작업이 완료되었습니다!")
    print(f"   통합된 데이터: {output_path}")
    print(f"   이제 강화학습 에이전트 학습에 사용할 수 있습니다.")

if __name__ == "__main__":
    main()

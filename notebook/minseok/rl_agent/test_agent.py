#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
환 리스크 헷지 강화학습 에이전트 테스트 스크립트
빠른 테스트 및 기능 검증용
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_environment():
    """환경 테스트"""
    print("🧪 환경 테스트 시작...")
    
    try:
        from hedge_environment import HedgeEnvironment
        
        # 샘플 데이터로 환경 생성
        env = HedgeEnvironment(
            data_path="sample_data.csv",  # 존재하지 않는 파일 (샘플 데이터 자동 생성)
            initial_capital=100000.0,
            max_hedge_ratio=0.8,
            transaction_cost=0.002,
            risk_free_rate=0.03
        )
        
        print(f"✅ 환경 생성 성공")
        print(f"   상태 차원: {env.observation_space.shape}")
        print(f"   액션 차원: {env.action_space.n}")
        print(f"   최대 스텝: {env.max_steps}")
        
        # 환경 리셋 테스트
        state, info = env.reset()
        print(f"✅ 환경 리셋 성공")
        print(f"   초기 상태 차원: {state.shape}")
        print(f"   초기 자본금: ${info['current_capital']:,.2f}")
        
        # 간단한 액션 테스트
        action = 0  # HOLD
        next_state, reward, done, truncated, info = env.step(action)
        print(f"✅ 액션 실행 성공")
        print(f"   보상: {reward:.4f}")
        print(f"   다음 상태 차원: {next_state.shape}")
        print(f"   현재 자본금: ${info['current_capital']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """에이전트 테스트"""
    print("\n🤖 에이전트 테스트 시작...")
    
    try:
        from hedge_agent import HedgeAgent
        
        # 에이전트 생성
        agent = HedgeAgent(
            state_dim=765,
            action_dim=9,
            learning_rate=1e-3,
            gamma=0.95,
            gae_lambda=0.9,
            clip_ratio=0.1,
            value_loss_coef=0.3,
            entropy_coef=0.02,
            max_grad_norm=0.3
        )
        
        print(f"✅ 에이전트 생성 성공")
        print(f"   디바이스: {agent.device}")
        print(f"   모델 파라미터 수: {sum(p.numel() for p in agent.model.parameters()):,}")
        
        # 간단한 액션 선택 테스트
        import numpy as np
        test_state = np.random.randn(765).astype(np.float32)
        action = agent.select_action(test_state, training=True)
        print(f"✅ 액션 선택 성공")
        print(f"   선택된 액션: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에이전트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integration():
    """데이터 통합 테스트"""
    print("\n📊 데이터 통합 테스트 시작...")
    
    try:
        from data_integration import DataIntegrator
        
        # 데이터 통합기 생성
        integrator = DataIntegrator()
        print(f"✅ 데이터 통합기 생성 성공")
        
        # 샘플 데이터 생성 테스트
        sample_yonju = integrator._generate_sample_yonju_data()
        sample_chaewon = integrator._generate_sample_chaewon_data()
        
        print(f"✅ 샘플 데이터 생성 성공")
        print(f"   yonju 샘플: {len(sample_yonju)}행")
        print(f"   chaewon 샘플: {len(sample_chaewon)}행")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n🚀 전체 파이프라인 테스트 시작...")
    
    try:
        # 1. 데이터 통합
        print("📊 1단계: 데이터 통합 테스트")
        from data_integration import DataIntegrator
        integrator = DataIntegrator()
        
        # 샘플 데이터로 통합 테스트
        yonju_sample = integrator._generate_sample_yonju_data()
        chaewon_sample = integrator._generate_sample_chaewon_data()
        
        # 통합
        integrated = integrator.integrate_data()
        training_data = integrator.create_training_data(integrated)
        
        print(f"✅ 데이터 통합 성공: {len(training_data)}행")
        
        # 2. 환경 생성
        print("🌍 2단계: 환경 생성 테스트")
        from hedge_environment import HedgeEnvironment
        
        # 임시 데이터 파일 생성
        temp_data_path = current_dir / "temp_test_data.csv"
        training_data.to_csv(temp_data_path, index=False)
        
        env = HedgeEnvironment(
            data_path=str(temp_data_path),
            initial_capital=50000.0,
            max_hedge_ratio=0.5
        )
        
        print(f"✅ 환경 생성 성공")
        
        # 3. 에이전트 생성
        print("🤖 3단계: 에이전트 생성 테스트")
        from hedge_agent import HedgeAgent
        
        agent = HedgeAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=1e-3
        )
        
        print(f"✅ 에이전트 생성 성공")
        
        # 4. 간단한 학습 테스트
        print("🎯 4단계: 간단한 학습 테스트")
        
        # 몇 개의 에피소드만 테스트
        for episode in range(3):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done or truncated or episode_length >= 10:  # 최대 10스텝
                    break
            
            print(f"   에피소드 {episode + 1}: 보상 {episode_reward:.2f}, 길이 {episode_length}")
        
        print(f"✅ 간단한 학습 테스트 성공")
        
        # 임시 파일 정리
        if temp_data_path.exists():
            temp_data_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ 전체 파이프라인 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """모듈 임포트 테스트"""
    print("📦 모듈 임포트 테스트 시작...")
    
    try:
        # 필요한 패키지들 임포트 테스트
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        
        print(f"✅ 기본 패키지 임포트 성공")
        print(f"   NumPy: {np.__version__}")
        print(f"   Pandas: {pd.__version__}")
        print(f"   PyTorch: {torch.__version__}")
        
        # 커스텀 모듈 임포트 테스트
        try:
            from hedge_environment import HedgeEnvironment
            print("✅ HedgeEnvironment 임포트 성공")
        except Exception as e:
            print(f"❌ HedgeEnvironment 임포트 실패: {e}")
            return False
        
        try:
            from hedge_agent import HedgeAgent
            print("✅ HedgeAgent 임포트 성공")
        except Exception as e:
            print(f"❌ HedgeAgent 임포트 실패: {e}")
            return False
        
        try:
            from data_integration import DataIntegrator
            print("✅ DataIntegrator 임포트 성공")
        except Exception as e:
            print(f"❌ DataIntegrator 임포트 실패: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 모듈 임포트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 환 리스크 헷지 강화학습 에이전트 테스트 시작\n")
    
    test_results = []
    
    # 0. 모듈 임포트 테스트
    test_results.append(("모듈 임포트", test_imports()))
    
    # 1. 환경 테스트
    test_results.append(("환경", test_environment()))
    
    # 2. 에이전트 테스트
    test_results.append(("에이전트", test_agent()))
    
    # 3. 데이터 통합 테스트
    test_results.append(("데이터 통합", test_data_integration()))
    
    # 4. 전체 파이프라인 테스트
    test_results.append(("전체 파이프라인", test_full_pipeline()))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📋 테스트 결과 요약")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n전체 테스트: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        print("이제 main.py를 실행하여 전체 파이프라인을 테스트할 수 있습니다.")
    else:
        print("⚠️  일부 테스트가 실패했습니다. 오류 메시지를 확인해 주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

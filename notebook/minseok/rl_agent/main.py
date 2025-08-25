#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
환 리스크 헷지 강화학습 에이전트 메인 실행 파일
데이터 통합, 학습, 평가의 전체 파이프라인 실행
"""

import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 필요한 모듈들 import
try:
    from data_integration import DataIntegrator
    from hedge_environment import HedgeEnvironment
    from hedge_agent import HedgeAgent
    print("✅ 모든 모듈 import 성공")
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("현재 디렉토리에 필요한 파일들이 있는지 확인해 주세요.")
    sys.exit(1)

def setup_environment():
    """환경 설정"""
    print("🔧 환경 설정 중...")
    
    # 출력 디렉토리 생성
    output_dir = current_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # 모델 저장 디렉토리 생성
    model_dir = current_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # 통합 데이터 디렉토리 생성
    integrated_data_dir = current_dir / "integrated_data"
    integrated_data_dir.mkdir(exist_ok=True)
    
    print(f"✅ 환경 설정 완료")
    print(f"   출력 디렉토리: {output_dir}")
    print(f"   모델 디렉토리: {model_dir}")
    print(f"   통합 데이터 디렉토리: {integrated_data_dir}")
    
    return output_dir, model_dir, integrated_data_dir

def run_data_integration():
    """데이터 통합 실행"""
    print("\n" + "="*60)
    print("📊 1단계: 데이터 통합")
    print("="*60)
    
    try:
        integrator = DataIntegrator()
        data_path = integrator.run_integration()
        return data_path
    except Exception as e:
        print(f"❌ 데이터 통합 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_training(data_path: str, output_dir: Path, model_dir: Path, 
                num_episodes: int = 1000, eval_interval: int = 100):
    """강화학습 에이전트 학습 실행"""
    print("\n" + "="*60)
    print("🎯 2단계: 강화학습 에이전트 학습")
    print("="*60)
    
    try:
        # 환경 생성
        print("🌍 강화학습 환경 생성 중...")
        env = HedgeEnvironment(
            data_path=data_path,
            initial_capital=1000000.0,  # 100만 달러
            max_hedge_ratio=1.0,        # 최대 헷지 비율 100%
            transaction_cost=0.001,     # 거래 비용 0.1%
            risk_free_rate=0.02         # 무위험 수익률 2%
        )
        
        # 에이전트 생성
        print("🤖 강화학습 에이전트 생성 중...")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = HedgeAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5
        )
        
        print(f"📊 환경 정보:")
        print(f"   상태 차원: {state_dim}")
        print(f"   액션 차원: {action_dim}")
        print(f"   최대 스텝: {env.max_steps}")
        
        # 학습 실행
        print(f"\n🚀 학습 시작: {num_episodes} 에피소드")
        training_history = agent.train(env, num_episodes, eval_interval)
        
        # 학습 결과 시각화
        print("\n📊 학습 결과 시각화 중...")
        plot_path = output_dir / "training_history.png"
        agent.plot_training_history(save_path=str(plot_path))
        
        # 모델 저장
        print("\n💾 모델 저장 중...")
        model_path = model_dir / "hedge_agent_model.pth"
        agent.save_model(str(model_path))
        
        return agent, training_history
        
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_evaluation(agent: HedgeAgent, data_path: str, output_dir: Path, 
                  num_eval_episodes: int = 100):
    """에이전트 평가 실행"""
    print("\n" + "="*60)
    print("🔍 3단계: 에이전트 평가")
    print("="*60)
    
    try:
        # 평가용 환경 생성
        print("🌍 평가용 환경 생성 중...")
        eval_env = HedgeEnvironment(
            data_path=data_path,
            initial_capital=1000000.0,
            max_hedge_ratio=1.0,
            transaction_cost=0.001,
            risk_free_rate=0.02
        )
        
        # 에이전트 평가
        evaluation_results = agent.evaluate(eval_env, num_eval_episodes)
        
        # 평가 결과 저장
        print("\n💾 평가 결과 저장 중...")
        eval_path = output_dir / "evaluation_results.txt"
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write("=== 환 리스크 헷지 강화학습 에이전트 평가 결과 ===\n\n")
            f.write(f"평가 에피소드 수: {num_eval_episodes}\n\n")
            
            f.write("=== 성능 지표 ===\n")
            f.write(f"평균 보상: {evaluation_results['mean_reward']:.4f} ± {evaluation_results['std_reward']:.4f}\n")
            f.write(f"평균 에피소드 길이: {evaluation_results['mean_length']:.2f}\n")
            f.write(f"평균 최종 자본: ${evaluation_results['mean_final_capital']:,.2f}\n")
            f.write(f"최종 자본 표준편차: ${evaluation_results['std_final_capital']:,.2f}\n")
            f.write(f"샤프 비율: {evaluation_results['sharpe_ratio']:.4f}\n")
        
        print(f"✅ 평가 결과 저장 완료: {eval_path}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_backtesting(agent: HedgeAgent, data_path: str, output_dir: Path):
    """백테스팅 실행"""
    print("\n" + "="*60)
    print("📈 4단계: 백테스팅")
    print("="*60)
    
    try:
        # 백테스팅용 환경 생성
        print("🌍 백테스팅용 환경 생성 중...")
        backtest_env = HedgeEnvironment(
            data_path=data_path,
            initial_capital=1000000.0,
            max_hedge_ratio=1.0,
            transaction_cost=0.001,
            risk_free_rate=0.02
        )
        
        # 백테스팅 실행
        print("📊 백테스팅 실행 중...")
        state, info = backtest_env.reset()
        
        backtest_results = {
            'dates': [],
            'capitals': [],
            'hedge_ratios': [],
            'positions': [],
            'actions': [],
            'rewards': []
        }
        
        step = 0
        while True:
            # 액션 선택 (평가 모드)
            action = agent.select_action(state, training=False)
            
            # 환경에서 스텝 실행
            next_state, reward, done, truncated, info = backtest_env.step(action)
            
            # 결과 저장
            if step < len(backtest_env.data):
                current_date = backtest_env.data.iloc[step]['date']
                backtest_results['dates'].append(current_date)
                backtest_results['capitals'].append(info['current_capital'])
                backtest_results['hedge_ratios'].append(info['hedge_ratio'])
                backtest_results['positions'].append(info['position'])
                backtest_results['actions'].append(action)
                backtest_results['rewards'].append(reward)
            
            # 상태 업데이트
            state = next_state
            step += 1
            
            if done or truncated:
                break
        
        # 백테스팅 결과 시각화
        print("📊 백테스팅 결과 시각화 중...")
        import matplotlib.pyplot as plt
        import pandas as pd
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 자본금 변화
        axes[0, 0].plot(backtest_results['dates'], backtest_results['capitals'])
        axes[0, 0].set_title('자본금 변화')
        axes[0, 0].set_xlabel('날짜')
        axes[0, 0].set_ylabel('자본금 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 헷지 비율 변화
        axes[0, 1].plot(backtest_results['dates'], backtest_results['hedge_ratios'])
        axes[0, 1].set_title('헷지 비율 변화')
        axes[0, 1].set_xlabel('날짜')
        axes[0, 1].set_ylabel('헷지 비율')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 액션 분포
        action_counts = pd.Series(backtest_results['actions']).value_counts()
        axes[1, 0].bar(range(len(action_counts)), action_counts.values)
        axes[1, 0].set_title('액션 분포')
        axes[1, 0].set_xlabel('액션')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].set_xticks(range(len(action_counts)))
        axes[1, 0].set_xticklabels([f'A{i}' for i in action_counts.index])
        
        # 보상 누적
        cumulative_rewards = np.cumsum(backtest_results['rewards'])
        axes[1, 1].plot(backtest_results['dates'], cumulative_rewards)
        axes[1, 1].set_title('누적 보상')
        axes[1, 1].set_xlabel('날짜')
        axes[1, 1].set_ylabel('누적 보상')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 백테스팅 결과 저장
        backtest_plot_path = output_dir / "backtest_results.png"
        plt.savefig(backtest_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 백테스팅 결과 저장: {backtest_plot_path}")
        
        # 백테스팅 데이터 저장
        backtest_df = pd.DataFrame(backtest_results)
        backtest_csv_path = output_dir / "backtest_results.csv"
        backtest_df.to_csv(backtest_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 백테스팅 데이터 저장: {backtest_csv_path}")
        
        plt.show()
        
        return backtest_results
        
    except Exception as e:
        print(f"❌ 백테스팅 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_test():
    """간단한 테스트 실행"""
    print("\n" + "="*60)
    print("🧪 테스트 모드 실행")
    print("="*60)
    
    try:
        # 1. 모듈 임포트 테스트
        print("📦 모듈 임포트 테스트...")
        from data_integration import DataIntegrator
        from hedge_environment import HedgeEnvironment
        from hedge_agent import HedgeAgent
        print("✅ 모든 모듈 임포트 성공")
        
        # 2. 데이터 통합 테스트
        print("\n📊 데이터 통합 테스트...")
        integrator = DataIntegrator()
        integrated = integrator.integrate_data()
        training_data = integrator.create_training_data(integrated)
        print(f"✅ 데이터 통합 테스트 성공: {len(training_data)}행")
        
        # 3. 환경 생성 테스트
        print("\n🌍 환경 생성 테스트...")
        temp_data_path = current_dir / "temp_test_data.csv"
        training_data.to_csv(temp_data_path, index=False)
        
        env = HedgeEnvironment(
            data_path=str(temp_data_path),
            initial_capital=50000.0,
            max_hedge_ratio=0.5
        )
        print("✅ 환경 생성 테스트 성공")
        
        # 4. 에이전트 생성 테스트
        print("\n🤖 에이전트 생성 테스트...")
        agent = HedgeAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=1e-3
        )
        print("✅ 에이전트 생성 테스트 성공")
        
        # 5. 간단한 학습 테스트
        print("\n🎯 간단한 학습 테스트...")
        for episode in range(2):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done or truncated or episode_length >= 5:  # 최대 5스텝
                    break
            
            print(f"   에피소드 {episode + 1}: 보상 {episode_reward:.2f}, 길이 {episode_length}")
        
        print("✅ 간단한 학습 테스트 성공")
        
        # 임시 파일 정리
        if temp_data_path.exists():
            temp_data_path.unlink()
        
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='환 리스크 헷지 강화학습 에이전트')
    parser.add_argument('--mode', choices=['all', 'data', 'train', 'eval', 'backtest', 'test'], 
                       default='test', help='실행 모드')
    parser.add_argument('--episodes', type=int, default=1000, help='학습 에피소드 수')
    parser.add_argument('--eval-episodes', type=int, default=100, help='평가 에피소드 수')
    parser.add_argument('--data-path', type=str, help='데이터 파일 경로 (기본값: 자동 생성)')
    
    args = parser.parse_args()
    
    print("🚀 환 리스크 헷지 강화학습 에이전트 시작!")
    print(f"📋 실행 모드: {args.mode}")
    
    # 환경 설정
    output_dir, model_dir, integrated_data_dir = setup_environment()
    
    # 데이터 경로 설정
    if args.data_path:
        data_path = args.data_path
        print(f"📁 사용자 지정 데이터 경로: {data_path}")
    else:
        data_path = None
    
    try:
        if args.mode == 'test':
            # 테스트 모드
            success = run_test()
            if success:
                print("\n✅ 테스트가 성공적으로 완료되었습니다!")
                print("이제 다른 모드로 실행할 수 있습니다:")
                print("  python main.py --mode data      # 데이터 통합만")
                print("  python main.py --mode train     # 학습만")
                print("  python main.py --mode all        # 전체 파이프라인")
            else:
                print("\n❌ 테스트가 실패했습니다. 오류를 확인해 주세요.")
                return
        
        if args.mode in ['all', 'data']:
            # 1단계: 데이터 통합
            data_path = run_data_integration()
            if data_path is None:
                print("❌ 데이터 통합 실패로 인해 중단됩니다.")
                return
        
        if args.mode in ['all', 'train']:
            # 2단계: 강화학습 에이전트 학습
            agent, training_history = run_training(
                data_path, output_dir, model_dir, 
                args.episodes, 100
            )
            if agent is None:
                print("❌ 학습 실패로 인해 중단됩니다.")
                return
        
        if args.mode in ['all', 'eval'] and 'agent' in locals():
            # 3단계: 에이전트 평가
            evaluation_results = run_evaluation(
                agent, data_path, output_dir, args.eval_episodes
            )
        
        if args.mode in ['all', 'backtest'] and 'agent' in locals():
            # 4단계: 백테스팅
            backtest_results = run_backtesting(agent, data_path, output_dir)
        
        if args.mode != 'test':
            print("\n" + "="*60)
            print("🎉 모든 작업이 성공적으로 완료되었습니다!")
            print("="*60)
            print(f"📁 결과 파일들:")
            print(f"   - 출력 디렉토리: {output_dir}")
            print(f"   - 모델 디렉토리: {model_dir}")
            print(f"   - 통합 데이터 디렉토리: {integrated_data_dir}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

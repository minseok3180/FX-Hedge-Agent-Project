#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KT 믿음 mini 모델을 사용한 외환 거래 AI 에이전트 시스템
"""

import os
import sys
import argparse
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from fx_data_collector import FXDataCollector
from kt_rag_system import KTRAGSystem
from fx_agent import FXAgent

def run_streamlit_app():
    """Streamlit 웹 앱 실행"""
    import subprocess
    import webbrowser
    import time
    
    print("🚀 Streamlit 웹 앱을 시작합니다...")
    
    # Streamlit 앱 실행
    app_path = current_dir / "streamlit_app.py"
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
    
    # 브라우저 자동 열기
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    
    print("🌐 웹 브라우저가 자동으로 열렸습니다.")
    print("📱 앱을 종료하려면 Ctrl+C를 누르세요.")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 앱을 종료합니다...")
        process.terminate()
        process.wait()

def run_console_app():
    """콘솔 기반 애플리케이션 실행"""
    print("🤖 KT Midm-2.0-Mini-Instruct 기반 외환 거래 AI 에이전트")
    print("=" * 50)
    
    try:
        # 시스템 초기화
        print("📡 데이터 수집기 초기화 중...")
        data_collector = FXDataCollector()
        
        print("🧠 KT Midm-2.0-Mini-Instruct 모델 RAG 시스템 초기화 중...")
        rag_system = KTRAGSystem()
        
        print("📚 외환 지식 베이스 구축 중...")
        rag_system.add_fx_knowledge_base()
        
        print("🤖 AI 에이전트 초기화 중...")
        fx_agent = FXAgent(rag_system, data_collector)
        
        print("✅ 시스템 초기화 완료!")
        print()
        
        # 메인 루프
        while True:
            print("\n" + "=" * 50)
            print("📋 메뉴 선택:")
            print("1. 📊 시장 분석")
            print("2. 🤖 AI 거래 추천")
            print("3. 📈 포트폴리오 현황")
            print("4. 💬 AI 상담")
            print("5. 🚀 Streamlit 웹 앱 실행")
            print("0. 🚪 종료")
            print("=" * 50)
            
            choice = input("선택하세요 (0-5): ").strip()
            
            if choice == "0":
                print("👋 시스템을 종료합니다. 안녕히 가세요!")
                break
                
            elif choice == "1":
                handle_market_analysis(fx_agent, data_collector)
                
            elif choice == "2":
                handle_trading_recommendation(fx_agent)
                
            elif choice == "3":
                handle_portfolio_status(fx_agent)
                
            elif choice == "4":
                handle_ai_consultation(rag_system)
                
            elif choice == "5":
                run_streamlit_app()
                break
                
            else:
                print("❌ 잘못된 선택입니다. 다시 시도해주세요.")
    
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")

def handle_market_analysis(fx_agent, data_collector):
    """시장 분석 처리"""
    print("\n📊 시장 분석")
    print("-" * 30)
    
    # 주요 통화쌍 목록 (KRW 기반)
    pairs = ["USDKRW=X", "EURKRW=X", "JPYKRW=X", "GBPKRW=X", "CNYKRW=X", "AUDKRW=X", "CADKRW=X", "CHFKRW=X"]
    
    # 통화쌍별 한국어 이름
    pair_names = {
        'USDKRW=X': '달러/원', 'EURKRW=X': '유로/원', 'JPYKRW=X': '엔/원', 'GBPKRW=X': '파운드/원',
        'CNYKRW=X': '위안/원', 'AUDKRW=X': '호주달러/원', 'CADKRW=X': '캐나다달러/원', 'CHFKRW=X': '스위스프랑/원'
    }
    
    print("분석할 통화쌍을 선택하세요:")
    for i, pair in enumerate(pairs, 1):
        pair_name = pair_names.get(pair, pair.replace('=X', ''))
        print(f"{i}. {pair_name}")
    
    try:
        choice = int(input("선택 (1-8): ")) - 1
        if 0 <= choice < len(pairs):
            selected_pair = pairs[choice]
            print(f"\n🔍 {selected_pair.replace('=X', '')} 분석 중...")
            
            analysis = fx_agent.analyze_market_condition(selected_pair)
            
            if "error" not in analysis:
                print(f"✅ 분석 완료!")
                print(f"현재 가격: {analysis['current_price']:.4f}")
                print(f"추세: {analysis['trend']}")
                print(f"RSI: {analysis['rsi_signal']}")
                print(f"MACD: {analysis['macd_signal']}")
                print(f"볼린저 밴드: {analysis['bb_position']}")
                print(f"변동성: {analysis['volatility']:.2f}%")
                print(f"지지선: {analysis['support_level']:.4f}")
                print(f"저항선: {analysis['resistance_level']:.4f}")
            else:
                print(f"❌ 분석 실패: {analysis['error']}")
        else:
            print("❌ 잘못된 선택입니다.")
    except ValueError:
        print("❌ 숫자를 입력해주세요.")

def handle_trading_recommendation(fx_agent):
    """거래 추천 처리"""
    print("\n🤖 AI 거래 추천")
    print("-" * 30)
    
    pairs = ["USDKRW=X", "EURKRW=X", "JPYKRW=X", "GBPKRW=X", "CNYKRW=X", "AUDKRW=X", "CADKRW=X", "CHFKRW=X"]
    
    # 통화쌍별 한국어 이름
    pair_names = {
        'USDKRW=X': '달러/원', 'EURKRW=X': '유로/원', 'JPYKRW=X': '엔/원', 'GBPKRW=X': '파운드/원',
        'CNYKRW=X': '위안/원', 'AUDKRW=X': '호주달러/원', 'CADKRW=X': '캐나다달러/원', 'CHFKRW=X': '스위스프랑/원'
    }
    
    print("추천을 받을 통화쌍을 선택하세요:")
    for i, pair in enumerate(pairs, 1):
        pair_name = pair_names.get(pair, pair.replace('=X', ''))
        print(f"{i}. {pair_name}")
    
    try:
        choice = int(input("선택 (1-8): ")) - 1
        if 0 <= choice < len(pairs):
            selected_pair = pairs[choice]
            print(f"\n🤖 {selected_pair.replace('=X', '')} AI 분석 중...")
            
            recommendation = fx_agent.generate_trading_recommendation(selected_pair)
            
            if "error" not in recommendation:
                print(f"✅ AI 추천 완료!")
                print(f"거래 신호: {recommendation['signal']}")
                print(f"신뢰도: {recommendation['confidence']:.1f}%")
                print(f"신호 강도: {recommendation['signal_strength']:.2f}")
                print(f"현재 가격: {recommendation['current_price']:.4f}")
                print(f"목표 가격: {recommendation['target_price']:.4f}")
                print(f"손절매 가격: {recommendation['stop_loss']:.4f}")
                print(f"이유: {recommendation['reasoning']}")
                print(f"\n🤖 AI 전문가 조언:")
                print(recommendation['expert_advice'])
                
                # 거래 실행 여부
                execute = input("\n거래를 실행하시겠습니까? (y/n): ").lower().strip()
                if execute == 'y':
                    try:
                        position_size = float(input("포지션 크기를 입력하세요 (0.1-10.0): "))
                        if 0.1 <= position_size <= 10.0:
                            result = fx_agent.execute_trade(recommendation, position_size)
                            print(f"✅ {result['message']}")
                        else:
                            print("❌ 포지션 크기는 0.1에서 10.0 사이여야 합니다.")
                    except ValueError:
                        print("❌ 올바른 숫자를 입력해주세요.")
            else:
                print(f"❌ 추천 생성 실패: {recommendation['error']}")
        else:
            print("❌ 잘못된 선택입니다.")
    except ValueError:
        print("❌ 숫자를 입력해주세요.")

def handle_portfolio_status(fx_agent):
    """포트폴리오 현황 처리"""
    print("\n📈 포트폴리오 현황")
    print("-" * 30)
    
    # 포트폴리오 요약
    portfolio = fx_agent.get_portfolio_summary()
    
    print(f"활성 포지션: {portfolio['active_positions']}")
    print(f"완료된 거래: {portfolio['closed_positions']}")
    print(f"총 손익: {portfolio['total_pnl']:.2f}")
    print(f"총 손익률: {portfolio['total_pnl_percentage']:.2f}%")
    print(f"평균 손익률: {portfolio['average_pnl_percentage']:.2f}%")
    print(f"승률: {portfolio['win_rate']*100:.1f}%")
    
    # 포지션 업데이트
    if portfolio['active_positions'] > 0:
        update = input("\n포지션을 업데이트하시겠습니까? (y/n): ").lower().strip()
        if update == 'y':
            print("🔄 포지션 업데이트 중...")
            updated = fx_agent.update_positions()
            print(f"✅ {len(updated)}개 포지션이 업데이트되었습니다.")
            
            # 업데이트된 포트폴리오 정보
            portfolio = fx_agent.get_portfolio_summary()
            print(f"\n업데이트된 포트폴리오:")
            print(f"활성 포지션: {portfolio['active_positions']}")
            print(f"총 손익: {portfolio['total_pnl']:.2f}")

def handle_ai_consultation(rag_system):
    """AI 상담 처리"""
    print("\n💬 AI 상담")
    print("-" * 30)
    
    question = input("외환 거래에 대한 질문을 입력하세요: ").strip()
    
    if question:
        print("\n🤖 AI가 답변을 생성하고 있습니다...")
        try:
            answer = rag_system.rag_query(question)
            print(f"\n✅ AI 답변:")
            print(answer)
        except Exception as e:
            print(f"❌ 답변 생성 중 오류가 발생했습니다: {e}")
    else:
        print("❌ 질문을 입력해주세요.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="KT Midm-2.0-Mini-Instruct 기반 외환 거래 AI 에이전트")
    parser.add_argument(
        "--mode", 
        choices=["console", "web"], 
        default="console",
        help="실행 모드 선택 (console: 콘솔, web: 웹 앱)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        run_streamlit_app()
    else:
        run_console_app()

if __name__ == "__main__":
    main()
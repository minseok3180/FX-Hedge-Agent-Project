#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KT 믿음 mini 기반 외환 거래 AI 에이전트 시스템 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_data_collector():
    """데이터 수집기 테스트"""
    print("🧪 데이터 수집기 테스트 시작...")
    
    try:
        from fx_data_collector import FXDataCollector
        
        collector = FXDataCollector()
        print("✅ FXDataCollector 초기화 성공")
        
        # 간단한 데이터 수집 테스트
        test_symbol = "EURUSD=X"
        data = collector.get_fx_data(test_symbol, "1mo")
        
        if not data.empty:
            print(f"✅ {test_symbol} 데이터 수집 성공 (데이터 수: {len(data)})")
            print(f"   최신 가격: {data['Close'].iloc[-1]:.4f}")
        else:
            print(f"❌ {test_symbol} 데이터 수집 실패")
            
    except Exception as e:
        print(f"❌ 데이터 수집기 테스트 실패: {e}")
        return False
    
    return True

def test_rag_system():
    """KT RAG 시스템 테스트"""
    print("\n🧪 KT RAG 시스템 테스트 시작...")
    
    try:
        from kt_rag_system import KTRAGSystem
        
        # KT RAG 시스템 초기화
        rag_system = KTRAGSystem()
        print("✅ KTRAGSystem 초기화 성공")
        
        # 지식 베이스 구축
        rag_system.add_fx_knowledge_base()
        
        # 간단한 질의응답 테스트
        test_query = "한국 시장에서 외환 거래 시 주의사항은 무엇인가요?"
        response = rag_system.rag_query(test_query)
        print(f"✅ RAG 질의응답 테스트 성공: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG 시스템 테스트 실패: {e}")
        return False

def test_fx_agent():
    """FX 에이전트 테스트"""
    print("\n🧪 FX 에이전트 테스트 시작...")
    
    try:
        from fx_agent import FXAgent
        from fx_data_collector import FXDataCollector
        
        # KT RAG 시스템 사용
        from kt_rag_system import KTRAGSystem
        
        collector = FXDataCollector()
        kt_rag = KTRAGSystem()
        kt_rag.add_fx_knowledge_base()
        agent = FXAgent(kt_rag, collector)
        
        print("✅ FXAgent 초기화 성공")
        
        # 포트폴리오 요약 테스트
        portfolio = agent.get_portfolio_summary()
        print(f"✅ 포트폴리오 요약 생성 성공: {portfolio}")
        
    except Exception as e:
        print(f"❌ FX 에이전트 테스트 실패: {e}")
        return False
    
    return True

def test_streamlit_imports():
    """Streamlit 관련 패키지 import 테스트"""
    print("\n🧪 Streamlit 패키지 테스트 시작...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import 성공")
        
        import plotly.graph_objects as go
        print("✅ Plotly import 성공")
        
        import pandas as pd
        print("✅ Pandas import 성공")
        
        import numpy as np
        print("✅ NumPy import 성공")
        
    except ImportError as e:
        print(f"❌ 패키지 import 실패: {e}")
        return False
    
    return True

def main():
    """메인 테스트 함수"""
    print("🚀 KT Midm-2.0-Mini-Instruct 기반 외환 거래 AI 에이전트 시스템 테스트")
    print("=" * 60)
    
    test_results = []
    
    # 각 모듈 테스트
    test_results.append(("데이터 수집기", test_data_collector()))
    test_results.append(("RAG 시스템", test_rag_system()))
    test_results.append(("FX 에이전트", test_fx_agent()))
    test_results.append(("Streamlit 패키지", test_streamlit_imports()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n전체 테스트: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 통과했습니다!")
        print("\n🚀 시스템을 실행할 준비가 되었습니다:")
        print("   콘솔 모드: python src/main.py")
        print("   웹 앱 모드: python src/main.py --mode web")
    else:
        print("⚠️  일부 테스트가 실패했습니다. requirements.txt를 확인하고 패키지를 설치해주세요.")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()

"""
외환 헷지전략 에이전트 Streamlit 앱
Mi:dm 2.0 모델을 활용한 멀티 에이전트 시스템 테스트 및 모니터링
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 모듈 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from supervisor import Supervisor
from agent.search_agent import SearchAgent
from agent.trading_agent import TradingAgent
from local_llm_client import LocalMidm2Client


def init_session_state():
    """세션 상태 초기화"""
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'supervisor' not in st.session_state:
        st.session_state.supervisor = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {}
    if 'llm_test_results' not in st.session_state:
        st.session_state.llm_test_results = {}


def load_configuration():
    """설정 로드 및 검증"""
    try:
        config = Config()
        
        # 설정 유효성 검증
        if config.validate_config():
            st.session_state.config = config
            return True
        else:
            st.error("설정 검증에 실패했습니다.")
            return False
            
    except Exception as e:
        st.error(f"설정 로드 중 오류: {str(e)}")
        return False


async def initialize_system():
    """시스템 초기화"""
    try:
        with st.spinner("시스템 초기화 중..."):
            # Supervisor 초기화
            supervisor = Supervisor(st.session_state.config)
            
            # 에이전트 등록 (실제 구현에서는 에이전트 인스턴스를 생성해야 함)
            # supervisor.agent_registry[AgentType.SEARCH] = SearchAgent(st.session_state.config)
            # supervisor.agent_registry[AgentType.TRADING] = TradingAgent(st.session_state.config)
            
            st.session_state.supervisor = supervisor
            
            # 시스템 상태 확인
            status = supervisor.get_system_status()
            st.session_state.system_status = status
            
        return True
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return False


async def test_llm_connection():
    """로컬 Mi:dm 2.0 LLM 연결 테스트"""
    try:
        with st.spinner("로컬 Mi:dm 2.0 모델 연결 테스트 중..."):
            client = LocalMidm2Client(st.session_state.config)
            
            # 간단한 테스트 메시지
            messages = [
                client.create_system_message("당신은 외환 헷지전략 전문가입니다."),
                client.create_user_message("안녕하세요. 외환 헷지전략에 대해 간단히 설명해주세요.")
            ]
            
            response = await client.chat_completion(messages, temperature=0.7)
            
            return {
                "success": True,
                "response": response.content,
                "model": response.model,
                "response_time": response.response_time,
                "usage": response.usage
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def test_embedding():
    """로컬 임베딩 모델 테스트"""
    try:
        with st.spinner("로컬 임베딩 모델 테스트 중..."):
            client = LocalMidm2Client(st.session_state.config)
            test_text = "외환 헷지전략 분석"
            embedding = await client.create_embedding(test_text)
            
            return {
                "success": True,
                "embedding_dimension": len(embedding),
                "sample_embedding": embedding[:5]  # 처음 5개 값만 표시
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def process_user_query(query: str):
    """사용자 질문 처리"""
    try:
        with st.spinner("질문 처리 중..."):
            result = await st.session_state.supervisor.process_user_query(
                user_id="streamlit_user",
                query=query,
                session_id="streamlit_session"
            )
            
            # 대화 히스토리에 추가
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_query": query,
                "response": result.get("response", ""),
                "status": result.get("status", "unknown"),
                "intent_analysis": result.get("intent_analysis", {})
            })
            
            return result
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def display_system_status():
    """시스템 상태 표시"""
    st.subheader("🔧 시스템 상태")
    
    if not st.session_state.system_status:
        st.warning("시스템 상태 정보가 없습니다.")
        return
    
    status = st.session_state.system_status
    
    # 상태 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="전체 상태",
            value="🟢 정상" if status.get("status") == "healthy" else "🔴 오류"
        )
    
    with col2:
        st.metric(
            label="활성 대화",
            value=status.get("active_conversations", 0)
        )
    
    with col3:
        agent_status = status.get("agent_status", {})
        available_agents = sum(1 for status in agent_status.values() if status == "available")
        st.metric(
            label="사용 가능 에이전트",
            value=f"{available_agents}/{len(agent_status)}"
        )
    
    with col4:
        llm_status = status.get("llm_status", {})
        st.metric(
            label="LLM 모델",
            value="🟢 연결됨" if llm_status.get("status") == "available" else "🔴 연결 안됨"
        )
    
    # 상세 정보
    with st.expander("상세 시스템 정보"):
        st.json(status)


def display_llm_test_results():
    """로컬 LLM 테스트 결과 표시"""
    st.subheader("🤖 로컬 Mi:dm 2.0 모델 테스트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("로컬 LLM 연결 테스트", type="primary"):
            result = asyncio.run(test_llm_connection())
            st.session_state.llm_test_results["connection"] = result
            
            if result["success"]:
                st.success("✅ 로컬 LLM 연결 성공!")
                st.text_area("모델 응답:", result["response"], height=200)
                
                # 응답 통계
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("응답 시간", f"{result['response_time']:.2f}초")
                with col_b:
                    st.metric("모델", result["model"])
                with col_c:
                    usage = result.get("usage", {})
                    st.metric("토큰 사용량", usage.get("total_tokens", 0))
            else:
                st.error(f"❌ 로컬 LLM 연결 실패: {result['error']}")
    
    with col2:
        if st.button("로컬 임베딩 모델 테스트"):
            result = asyncio.run(test_embedding())
            st.session_state.llm_test_results["embedding"] = result
            
            if result["success"]:
                st.success("✅ 로컬 임베딩 모델 연결 성공!")
                st.metric("임베딩 차원", result["embedding_dimension"])
                st.text_area("샘플 임베딩:", str(result["sample_embedding"]), height=100)
            else:
                st.error(f"❌ 로컬 임베딩 모델 연결 실패: {result['error']}")


def display_conversation_interface():
    """대화형 인터페이스"""
    st.subheader("💬 외환 헷지전략 에이전트와 대화하기")
    
    # 질문 입력
    user_query = st.text_area(
        "질문을 입력하세요:",
        placeholder="예: 현재 USD/KRW 환율 변동성에 대한 헷지 전략을 추천해주세요.",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("질문하기", type="primary", disabled=not user_query.strip()):
            if st.session_state.supervisor:
                result = asyncio.run(process_user_query(user_query))
                
                if result["status"] == "success":
                    st.success("질문이 처리되었습니다!")
                else:
                    st.error(f"처리 중 오류: {result.get('error', '알 수 없는 오류')}")
            else:
                st.error("시스템이 초기화되지 않았습니다.")
    
    with col2:
        if st.button("대화 히스토리 초기화"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # 대화 히스토리 표시
    if st.session_state.conversation_history:
        st.subheader("📝 대화 히스토리")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):  # 최근 5개만 표시
            with st.expander(f"질문 {len(st.session_state.conversation_history) - i}: {conv['user_query'][:50]}..."):
                st.write(f"**시간:** {conv['timestamp']}")
                st.write(f"**질문:** {conv['user_query']}")
                st.write(f"**응답:** {conv['response']}")
                
                if conv.get('intent_analysis'):
                    st.write("**의도 분석:**")
                    st.json(conv['intent_analysis'])


def display_configuration_info():
    """설정 정보 표시"""
    st.subheader("⚙️ 시스템 설정")
    
    if st.session_state.config:
        config_dict = st.session_state.config.to_dict()
        
        # LLM 설정
        st.write("**LLM 설정:**")
        llm_config = config_dict.get("llm", {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"- 모델: {llm_config.get('model_name', 'N/A')}")
            st.write(f"- 최대 토큰: {llm_config.get('max_tokens', 'N/A')}")
        
        with col2:
            st.write(f"- 온도: {llm_config.get('temperature', 'N/A')}")
            st.write(f"- 임베딩 모델: {llm_config.get('embedding_model', 'N/A')}")
        
        # 데이터베이스 설정
        st.write("**데이터베이스 설정:**")
        db_config = config_dict.get("database", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tsdb = db_config.get("tsdb", {})
            st.write(f"**TSDB:** {tsdb.get('host', 'N/A')}:{tsdb.get('port', 'N/A')}")
        
        with col2:
            vdb = db_config.get("vdb", {})
            st.write(f"**VDB:** {vdb.get('host', 'N/A')}:{vdb.get('port', 'N/A')}")
        
        with col3:
            rdb = db_config.get("rdb", {})
            st.write(f"**RDB:** {rdb.get('host', 'N/A')}:{rdb.get('port', 'N/A')}")
        
        # 전체 설정 보기
        with st.expander("전체 설정 보기"):
            st.json(config_dict)
    else:
        st.warning("설정이 로드되지 않았습니다.")


def main():
    """메인 함수"""
    st.set_page_config(
        page_title="외환 헷지전략 에이전트",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💰 외환 헷지전략 에이전트")
    st.markdown("**Mi:dm 2.0 모델을 활용한 멀티 에이전트 시스템**")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바
    with st.sidebar:
        st.header("🎛️ 제어판")
        
        if st.button("시스템 초기화", type="primary"):
            if load_configuration():
                if asyncio.run(initialize_system()):
                    st.success("✅ 시스템 초기화 완료!")
                else:
                    st.error("❌ 시스템 초기화 실패!")
            else:
                st.error("❌ 설정 로드 실패!")
        
        st.divider()
        
        # 네비게이션
        page = st.selectbox(
            "페이지 선택:",
            ["🏠 홈", "🤖 LLM 테스트", "💬 대화하기", "📊 시스템 상태", "⚙️ 설정"]
        )
    
    # 메인 콘텐츠
    if page == "🏠 홈":
        st.header("🏠 홈")
        st.markdown("""
        ### 외환 헷지전략 에이전트에 오신 것을 환영합니다!
        
        이 시스템은 KT의 **Mi:dm 2.0 모델**을 활용하여 외환 헷지전략을 분석하고 거래를 지원하는 멀티 에이전트 시스템입니다.
        
        #### 주요 기능:
        - 🔍 **Search Agent**: 헷지 관련 뉴스 검색 및 시계열 데이터 조회
        - 💼 **Trading Agent**: RAG 기반 헷지 전략 분석 및 거래 실행
        - 🎯 **Supervisor**: 질문 분석 및 에이전트 라우팅
        
        #### 사용 방법:
        1. 왼쪽 사이드바에서 "시스템 초기화" 버튼을 클릭하세요
        2. "LLM 테스트" 페이지에서 Mi:dm 2.0 모델 연결을 확인하세요
        3. "대화하기" 페이지에서 질문을 입력하세요
        """)
        
        # 시스템 상태 요약
        if st.session_state.system_status:
            st.success("✅ 시스템이 정상적으로 실행 중입니다!")
        else:
            st.warning("⚠️ 시스템을 초기화해주세요.")
    
    elif page == "🤖 LLM 테스트":
        display_llm_test_results()
    
    elif page == "💬 대화하기":
        display_conversation_interface()
    
    elif page == "📊 시스템 상태":
        display_system_status()
    
    elif page == "⚙️ 설정":
        display_configuration_info()
    
    # 푸터
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        외환 헷지전략 에이전트 v1.0 | KT Mi:dm 2.0 모델 기반
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# 상대 경로 import를 위한 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fx_data_collector import FXDataCollector
from kt_rag_system import KTRAGSystem
from fx_agent import FXAgent

def main():
    st.set_page_config(
        page_title="외환 거래 AI 에이전트",
        page_icon="💱",
        layout="wide"
    )
    
    st.title("💱 KT Midm-2.0-Mini-Instruct 기반 외환 거래 AI 에이전트")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    
    # 모델 로딩 상태 확인
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    if not st.session_state.system_initialized:
        with st.spinner("시스템 초기화 중..."):
            try:
                # 시스템 초기화
                st.session_state.data_collector = FXDataCollector()
                st.session_state.rag_system = KTRAGSystem()
                st.session_state.rag_system.add_fx_knowledge_base()
                st.session_state.fx_agent = FXAgent(
                    st.session_state.rag_system,
                    st.session_state.data_collector
                )
                st.session_state.system_initialized = True
                st.success("시스템 초기화 완료!")
            except Exception as e:
                st.error(f"시스템 초기화 실패: {e}")
                st.stop()
    
    # 메인 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📊 시장 분석", "🤖 AI 거래 추천", "📈 포트폴리오", "💬 AI 상담"])
    
    with tab1:
        st.header("📊 실시간 시장 분석")
        
        # 통화쌍 선택
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_pair = st.selectbox(
                "통화쌍 선택",
                ["USDKRW=X", "EURKRW=X", "JPYKRW=X", "GBPKRW=X", "CNYKRW=X", "AUDKRW=X", "CADKRW=X", "CHFKRW=X"],
                format_func=lambda x: {"USDKRW=X": "달러/원", "EURKRW=X": "유로/원", "JPYKRW=X": "엔/원", 
                                     "GBPKRW=X": "파운드/원", "CNYKRW=X": "위안/원", "AUDKRW=X": "호주달러/원", 
                                     "CADKRW=X": "캐나다달러/원", "CHFKRW=X": "스위스프랑/원"}.get(x, x.replace("=X", ""))
            )
            
            if st.button("분석 실행"):
                with st.spinner("시장 분석 중..."):
                    analysis = st.session_state.fx_agent.analyze_market_condition(selected_pair)
                    
                    if "error" not in analysis:
                        st.session_state.current_analysis = analysis
                        st.success("분석 완료!")
                    else:
                        st.error(analysis["error"])
        
        # 분석 결과 표시
        if 'current_analysis' in st.session_state:
            analysis = st.session_state.current_analysis
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("현재 가격", f"{analysis['current_price']:.4f}")
                st.metric("추세", analysis['trend'])
            
            with col2:
                st.metric("RSI", f"{analysis['rsi_signal']}")
                st.metric("MACD", analysis['macd_signal'])
            
            with col3:
                st.metric("볼린저 밴드", analysis['bb_position'])
                st.metric("변동성", f"{analysis['volatility']:.2f}%")
            
            with col4:
                st.metric("지지선", f"{analysis['support_level']:.4f}")
                st.metric("저항선", f"{analysis['resistance_level']:.4f}")
            
            # 차트 표시
            st.subheader("가격 차트")
            data = st.session_state.data_collector.get_fx_data(selected_pair, "3mo")
            if not data.empty:
                data = st.session_state.data_collector.calculate_technical_indicators(data)
                
                fig = go.Figure()
                
                # 캔들스틱 차트
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="가격"
                ))
                
                # 이동평균선
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    name="MA 20",
                    line=dict(color='orange')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    name="MA 50",
                    line=dict(color='blue')
                ))
                
                # 볼린저 밴드
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_upper'],
                    name="BB 상단",
                    line=dict(color='gray', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_lower'],
                    name="BB 하단",
                    line=dict(color='gray', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{selected_pair.replace('=X', '')} 차트",
                    xaxis_title="날짜",
                    yaxis_title="가격",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🤖 AI 거래 추천")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("거래 추천 요청")
            
            recommendation_pair = st.selectbox(
                "통화쌍 선택",
                ["USDKRW=X", "EURKRW=X", "JPYKRW=X", "GBPKRW=X", "CNYKRW=X", "AUDKRW=X", "CADKRW=X", "CHFKRW=X"],
                format_func=lambda x: {"USDKRW=X": "달러/원", "EURKRW=X": "유로/원", "JPYKRW=X": "엔/원", 
                                     "GBPKRW=X": "파운드/원", "CNYKRW=X": "위안/원", "AUDKRW=X": "호주달러/원", 
                                     "CADKRW=X": "캐나다달러/원", "CHFKRW=X": "스위스프랑/원"}.get(x, x.replace("=X", "")),
                key="recommendation_pair"
            )
            
            if st.button("AI 추천 생성"):
                with st.spinner("AI 분석 중..."):
                    recommendation = st.session_state.fx_agent.generate_trading_recommendation(recommendation_pair)
                    
                    if "error" not in recommendation:
                        st.session_state.current_recommendation = recommendation
                        st.success("AI 추천 완료!")
                    else:
                        st.error(recommendation["error"])
        
        # 추천 결과 표시
        if 'current_recommendation' in st.session_state:
            rec = st.session_state.current_recommendation
            
            with col2:
                st.subheader("AI 추천 결과")
                
                # 신호 강도 시각화
                signal_color = "green" if "매수" in rec["signal"] else "red" if "매도" in rec["signal"] else "gray"
                st.markdown(f"**거래 신호:** <span style='color:{signal_color}; font-size:24px;'>{rec['signal']}</span>", unsafe_allow_html=True)
                
                st.metric("신뢰도", f"{rec['confidence']:.1f}%")
                st.metric("신호 강도", f"{rec['signal_strength']:.2f}")
                
                st.info(f"**이유:** {rec['reasoning']}")
                
                # 거래 실행
                if st.button("거래 실행 (시뮬레이션)"):
                    position_size = st.number_input("포지션 크기", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    
                    if st.button("확정 실행"):
                        result = st.session_state.fx_agent.execute_trade(rec, position_size)
                        st.success(result["message"])
                        st.json(result["trade_details"])
            
            # 상세 정보
            st.subheader("상세 정보")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("현재 가격", f"{rec['current_price']:.4f}")
                st.metric("목표 가격", f"{rec['target_price']:.4f}")
            
            with col2:
                st.metric("손절매 가격", f"{rec['stop_loss']:.4f}")
                st.metric("예상 수익률", f"{(rec['target_price']/rec['current_price'] - 1) * 100:.2f}%")
            
            with col3:
                st.metric("리스크 대비 보상", f"{(rec['target_price'] - rec['current_price']) / (rec['current_price'] - rec['stop_loss']):.2f}")
            
            # AI 전문가 조언
            st.subheader("🤖 AI 전문가 조언")
            st.write(rec['expert_advice'])
    
    with tab3:
        st.header("📈 포트폴리오 현황")
        
        # 포트폴리오 요약
        portfolio = st.session_state.fx_agent.get_portfolio_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("활성 포지션", portfolio['active_positions'])
            st.metric("완료된 거래", portfolio['closed_positions'])
        
        with col2:
            st.metric("총 손익", f"{portfolio['total_pnl']:.2f}")
            st.metric("총 손익률", f"{portfolio['total_pnl_percentage']:.2f}%")
        
        with col3:
            st.metric("평균 손익률", f"{portfolio['average_pnl_percentage']:.2f}%")
            st.metric("승률", f"{portfolio['win_rate']*100:.1f}%")
        
        with col4:
            if st.button("포지션 업데이트"):
                with st.spinner("포지션 업데이트 중..."):
                    updated = st.session_state.fx_agent.update_positions()
                    st.success(f"{len(updated)}개 포지션이 업데이트되었습니다.")
        
        # 활성 포지션 표시
        if portfolio['active_positions'] > 0:
            st.subheader("활성 포지션")
            
            active_positions = [p for p in st.session_state.fx_agent.current_positions.values() if p["status"] == "진행중"]
            
            if active_positions:
                df = pd.DataFrame(active_positions)
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['entry_time'] = df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(df[['symbol', 'signal', 'entry_price', 'current_price', 'pnl', 'pnl_percentage', 'entry_time']], use_container_width=True)
        
        # 거래 히스토리
        if portfolio['closed_positions'] > 0:
            st.subheader("거래 히스토리")
            
            closed_positions = [p for p in st.session_state.fx_agent.trading_history if p["status"] != "진행중"]
            
            if closed_positions:
                df = pd.DataFrame(closed_positions)
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['entry_time'] = df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(df[['symbol', 'signal', 'entry_price', 'target_price', 'stop_loss', 'pnl', 'pnl_percentage', 'status', 'entry_time']], use_container_width=True)
    
    with tab4:
        st.header("💬 AI 상담")
        
        st.markdown("외환 거래에 대한 질문을 자유롭게 해주세요. KT Midm-2.0-Mini-Instruct 모델이 전문가 조언을 제공합니다.")
        
        # 질문 입력
        user_question = st.text_area(
            "질문을 입력하세요:",
            placeholder="예: EUR/USD의 현재 시장 상황은 어떻나요?",
            height=100
        )
        
        if st.button("AI 상담 받기"):
            if user_question.strip():
                with st.spinner("AI가 답변을 생성하고 있습니다..."):
                    try:
                        # RAG 시스템을 통한 답변
                        answer = st.session_state.rag_system.rag_query(user_question)
                        
                        st.subheader("🤖 AI 답변")
                        st.write(answer)
                        
                        # 답변 저장
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': answer,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                    except Exception as e:
                        st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
            else:
                st.warning("질문을 입력해주세요.")
        
        # 채팅 히스토리
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.subheader("💬 상담 히스토리")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"상담 {len(st.session_state.chat_history) - i} - {chat['timestamp']}"):
                    st.markdown(f"**질문:** {chat['question']}")
                    st.markdown(f"**답변:** {chat['answer']}")
    
    # 자동 새로고침
    if st.sidebar.checkbox("자동 새로고침", value=False):
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

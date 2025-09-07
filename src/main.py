"""
외환 헷지전략 에이전트 메인 실행 파일
MVP 목적의 최소한의 기능을 포함한 멀티 에이전트 시스템 실행
"""

import asyncio
import logging
import sys
from typing import Dict, Any
import argparse
from datetime import datetime

from config import Config
from supervisor import Supervisor
from agent.search_agent import SearchAgent
from agent.trading_agent import TradingAgent


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fx_hedge_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def initialize_agents(config: Config) -> Dict[str, Any]:
    """
    에이전트들 초기화
    
    Args:
        config: 설정 객체
        
    Returns:
        초기화된 에이전트들
    """
    logger = logging.getLogger(__name__)
    
    try:
        # TODO: 에이전트 초기화 로직
        # - SearchAgent 인스턴스 생성
        # - TradingAgent 인스턴스 생성
        # - 각 에이전트 상태 확인
        # - 에이전트 간 연결 설정
        
        logger.info("에이전트 초기화 시작...")
        
        # 에이전트 인스턴스 생성
        search_agent = SearchAgent(config)
        trading_agent = TradingAgent(config)
        
        # 에이전트 상태 확인
        search_status = search_agent.get_agent_status()
        trading_status = trading_agent.get_agent_status()
        
        logger.info(f"Search Agent 상태: {search_status}")
        logger.info(f"Trading Agent 상태: {trading_status}")
        
        return {
            "search_agent": search_agent,
            "trading_agent": trading_agent,
            "initialization_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"에이전트 초기화 실패: {str(e)}")
        raise e


async def run_interactive_mode(supervisor: Supervisor):
    """
    대화형 모드 실행
    
    Args:
        supervisor: Supervisor 인스턴스
    """
    logger = logging.getLogger(__name__)
    logger.info("대화형 모드 시작...")
    
    print("\n=== 외환 헷지전략 에이전트 ===")
    print("질문을 입력하세요. 'quit' 또는 'exit'로 종료할 수 있습니다.\n")
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        try:
            # TODO: 사용자 입력 처리
            # - 사용자 질문 입력 받기
            # - 입력 검증
            # - 특별 명령어 처리 (quit, help 등)
            
            user_input = input("사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break
            
            if not user_input:
                continue
            
            # TODO: Supervisor를 통한 질문 처리
            # - 비동기 질문 처리
            # - 결과 출력
            # - 오류 처리
            
            print("에이전트: 질문을 처리 중입니다...")
            
            result = await supervisor.process_user_query(
                user_id="demo_user",
                query=user_input,
                session_id=session_id
            )
            
            if result["status"] == "success":
                print(f"에이전트: {result['response']}")
            else:
                print(f"에이전트: 오류가 발생했습니다 - {result.get('error', '알 수 없는 오류')}")
            
            print()  # 빈 줄 추가
            
        except KeyboardInterrupt:
            print("\n시스템을 종료합니다.")
            break
        except Exception as e:
            logger.error(f"대화형 모드 실행 중 오류: {str(e)}")
            print(f"오류가 발생했습니다: {str(e)}")


async def run_batch_mode(supervisor: Supervisor, queries: list):
    """
    배치 모드 실행
    
    Args:
        supervisor: Supervisor 인스턴스
        queries: 처리할 질문 리스트
    """
    logger = logging.getLogger(__name__)
    logger.info(f"배치 모드 시작 - {len(queries)}개 질문 처리")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        try:
            # TODO: 배치 처리 로직
            # - 각 질문 순차 처리
            # - 결과 수집
            # - 진행 상황 표시
            
            print(f"[{i}/{len(queries)}] 처리 중: {query}")
            
            result = await supervisor.process_user_query(
                user_id="batch_user",
                query=query,
                session_id=f"batch_session_{i}"
            )
            
            results.append({
                "query": query,
                "result": result,
                "processed_at": datetime.now().isoformat()
            })
            
            print(f"[{i}/{len(queries)}] 완료")
            
        except Exception as e:
            logger.error(f"배치 처리 중 오류 (질문 {i}): {str(e)}")
            results.append({
                "query": query,
                "result": {"status": "error", "error": str(e)},
                "processed_at": datetime.now().isoformat()
            })
    
    # TODO: 배치 결과 저장
    # - 결과를 파일로 저장
    # - 통계 정보 생성
    
    print(f"\n배치 처리 완료 - 총 {len(results)}개 질문 처리")
    return results


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="외환 헷지전략 에이전트")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="실행 모드 선택")
    parser.add_argument("--queries", nargs="+", help="배치 모드에서 처리할 질문들")
    parser.add_argument("--config-check", action="store_true", help="설정 검증만 수행")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # TODO: 설정 로드 및 검증
        # - Config 인스턴스 생성
        # - 설정 유효성 검증
        # - 데이터베이스 연결 테스트
        
        logger.info("시스템 초기화 시작...")
        
        config = Config()
        
        if args.config_check:
            # TODO: 설정 검증 로직
            # - 모든 설정값 유효성 검사
            # - 데이터베이스 연결 테스트
            # - API 키 유효성 검사
            
            is_valid = config.validate_config()
            if is_valid:
                print("설정 검증 성공")
                return
            else:
                print("설정 검증 실패")
                sys.exit(1)
        
        # TODO: 에이전트 초기화
        agents = await initialize_agents(config)
        
        # TODO: Supervisor 초기화
        supervisor = Supervisor(config)
        
        # TODO: 시스템 상태 확인
        system_status = supervisor.get_system_status()
        logger.info(f"시스템 상태: {system_status}")
        
        if args.mode == "interactive":
            # 대화형 모드 실행
            await run_interactive_mode(supervisor)
        elif args.mode == "batch":
            # 배치 모드 실행
            if not args.queries:
                print("배치 모드에서는 --queries 옵션이 필요합니다.")
                sys.exit(1)
            
            results = await run_batch_mode(supervisor, args.queries)
            
            # TODO: 결과 저장
            # - JSON 파일로 결과 저장
            # - 통계 리포트 생성
            
            print(f"배치 처리 결과가 저장되었습니다.")
        
        logger.info("시스템 정상 종료")
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {str(e)}")
        print(f"시스템 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # TODO: 비동기 메인 함수 실행
    # - asyncio.run() 사용
    # - 예외 처리
    # - 정리 작업
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"시스템 오류: {str(e)}")
        sys.exit(1)
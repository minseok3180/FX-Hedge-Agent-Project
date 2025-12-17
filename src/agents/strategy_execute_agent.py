"""
전략 실행 Agent

환헷지 전략을 수행하는 에이전트입니다.
- 뉴스 센티멘트 분석 결과 → signal_score
- RDB (eiExchangeRate 테이블) → 환율 변동성(sigma_fx), 상관계수(rho)
- sigma_asset, lambda(위험회피도)는 임의 값 또는 사용자 정보에서 가져옴
- 위 값들을 이용해서 최적 환헷지 비중 w_H* 계산
"""

from typing import Optional, Dict, Any
import sys
import importlib.util
from pathlib import Path
import ast
import re

# 프로젝트 루트를 Python 경로에 추가 (직접 실행 시)
# import 전에 실행되어야 함
_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# .env 파일 로드 (직접 실행 시)
if __name__ == "__main__":
    from dotenv import load_dotenv
    env_path = _project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

from src.utils.agents import BaseAgent
# 계산 로직은 CalculatorTool의 메서드를 직접 사용 (tool wrapper가 아니라 실제 함수 호출)
from src.tools.calculator import CalculatorTool
from src.utils.logger import get_logger
from src.utils.llm import call_gpt
import json

logger = get_logger("strategy-execute-agent")

# news_sentimental_analysis를 직접 로드
news_sent_path = Path(__file__).parent.parent / "tools" / "news_sentimental_analysis.py"
spec = importlib.util.spec_from_file_location("news_sentimental_analysis", news_sent_path)
news_sent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(news_sent_module)
run_sentiment_analysis = news_sent_module.run_sentiment_analysis


STRATEGY_EXECUTE_SYSTEM_PROMPT = """당신은 환헷지 전략을 수행하는 전문 에이전트입니다.

## 주요 역할
1. 뉴스 센티멘트 분석을 통해 환율 시그널 점수(signal_score)를 추출
2. RDB에서 환율 데이터를 조회하여 변동성(sigma_fx)과 상관계수(rho) 계산
3. 사용자 정보에서 위험회피도(risk_aversion) 가져오기
4. 최적 환헷지 비중(w_H*) 계산 및 전략 실행

## 사용 가능한 도구
- compute_expected_fx_return: 환율 기대수익률 계산
- compute_optimal_hedge_weight: 최적 환헷지 비중 계산
- compute_fx_vol_and_rho_from_rdb: RDB에서 환율 변동성 및 상관계수 계산

## 응답 형식
계산 결과를 명확하고 구조화된 형식으로 제공하세요.
"""


def get_user_risk_aversion(user_id: Optional[str] = None) -> float:
    """
    RDB의 user_info 테이블에서 user_risk_aversion 값을 가져옵니다.
    
    Args:
        user_id: 사용자 ID. None이면 기본값 4.0 반환
        
    Returns:
        user_risk_aversion 값 (없으면 4.0)
    """
    if user_id is None:
        return 4.0
    
    try:
        import pymysql
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        db_host = os.getenv("DATABASE_HOST") or os.getenv("DB_HOST")
        db_port = int(os.getenv("DATABASE_PORT") or os.getenv("DB_PORT", 3306))
        db_user = os.getenv("DATABASE_USER") or os.getenv("DB_USER")
        db_password = os.getenv("DATABASE_PASSWORD") or os.getenv("DB_PASSWORD")
        db_name = os.getenv("DATABASE_NAME") or os.getenv("DB_NAME")
        
        if not all([db_host, db_user, db_password, db_name]):
            logger.warning("⚠️  데이터베이스 연결 정보가 없어 기본값 4.0을 사용합니다.")
            return 4.0
        
        # DB 연결
        conn = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        
        # user_risk_aversion 조회
        query = """
            SELECT user_risk_aversion
            FROM user_info
            WHERE user_id = %s
            LIMIT 1
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
        
        conn.close()
        
        if result and result.get("user_risk_aversion") is not None:
            risk_aversion = float(result["user_risk_aversion"])
            logger.info(f"✅ 사용자 {user_id}의 risk_aversion: {risk_aversion}")
            return risk_aversion
        else:
            logger.warning(f"⚠️  사용자 {user_id}의 risk_aversion 정보가 없어 기본값 4.0을 사용합니다.")
            return 4.0
            
    except ImportError:
        logger.warning("⚠️  pymysql이 설치되지 않아 기본값 4.0을 사용합니다.")
        return 4.0
    except Exception as e:
        logger.warning(f"⚠️  RDB에서 risk_aversion을 가져오는 중 오류 발생: {e}")
        logger.warning("   기본값 4.0을 사용합니다.")
        return 4.0


class StrategyExecuteAgent(BaseAgent):
    """환헷지 전략을 수행하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="strategy_execute",
            description="환헷지 전략을 수행하는 전문 에이전트"
        )
        self.calculator = CalculatorTool()
    
    async def _generate_answer(
        self,
        task: str,
        signal_score: float,
        expected_fx_return: float,
        sigma_asset: float,
        sigma_fx: float,
        rho: float,
        risk_aversion: float,
        alpha: float,
        w_H_star: Optional[float],
        w_UH_star: Optional[float],
    ) -> str:
        """
        Tool 결과를 바탕으로 LLM을 사용하여 자연어 답변 생성
        
        Args:
            task: 사용자 요청/작업 설명
            signal_score: 환율 시그널 점수
            expected_fx_return: 기대 환율 수익률
            sigma_asset: 자산 변동성
            sigma_fx: 환율 변동성
            rho: 상관계수
            risk_aversion: 위험회피도
            alpha: 스케일링 파라미터
            w_H_star: 최적 환헷지 비중
            w_UH_star: 비헷지 비중
            
        Returns:
            자연어 답변 문자열
        """
        try:
            # 계산 결과가 없으면 에러 메시지 반환
            if w_H_star is None:
                return "⚠️ 최적 환헷지 비중을 계산할 수 없습니다. 입력 파라미터를 확인해주세요."
            
            # Tool 결과를 구조화된 데이터로 정리
            calculation_results = {
                "signal_score": signal_score,
                "expected_fx_return": expected_fx_return,
                "sigma_asset": sigma_asset,
                "sigma_fx": sigma_fx,
                "rho": rho,
                "risk_aversion": risk_aversion,
                "alpha": alpha,
                "w_H_star": w_H_star,
                "w_UH_star": w_UH_star,
            }
            
            # System prompt
            system_prompt = """당신은 환헷지(FX hedging) 전략 전문가입니다. 아래 계산 결과(calculation_results)는 이미 산출된 사실(facts)이며 숫자를 임의로 바꾸지 마세요.

## 출력 목표
- 보고서/목차형(예: 1., 2., 3. / #### 섹션)으로 쓰지 말고, '짧은 요약(Summary)' 형태로 작성하세요.
- 사용자의 요청(task)에 직접 답하세요. (가장 중요한 결론을 첫 문장에 제시)
- w_H_star(환헷지 비율)과 w_UH_star(환노출 비율)을 반드시 포함하세요.
- signal_score와 expected_fx_return(E[R_FX])을 한 문장으로 해석하세요. (달러 강세/약세/중립)
- sigma_fx(환율 변동성), rho(상관계수), risk_aversion(lambda)가 결과에 미친 영향을 각 1문장 이내로 설명하세요.
- 결과가 경계값(boundary)인 경우(예: w_H_star가 0 또는 1에 매우 가까움) clip(상한/하한 제한) 가능성을 경고로 1문장 포함하세요.
- 과도한 확신을 피하고 '예상/가능성/모형 기준' 표현을 사용하세요.

## 형식 규칙
- Markdown은 사용하되, 최대 6줄 이내의 짧은 단락으로 끝내세요.
- 숫자는 보기 좋게 변환하세요: 
  - w_H_star, w_UH_star는 %로 표시 (예: 65%)
  - expected_fx_return은 소수와 %를 함께 표시 (예: -0.00007, -0.007%)
  - 변동성(sigma)은 %로 표시
"""
            
            # User prompt - 계산 결과를 JSON 형식으로 제공
            calculation_summary = json.dumps(calculation_results, ensure_ascii=False, indent=2)
            
            user_prompt = f"""사용자 요청: {task}

## 계산 결과
{calculation_summary}

요구사항(반드시 지킬 것):
- 답변 첫 문장에 결론: "권장 환헷지 비율 w_H_star=XX%, 환노출 비율 w_UH_star=YY%"를 포함
- 근거는 3개만 아주 짧게: (1) 환율 전망(expected_fx_return/ signal_score) (2) 환율 리스크(sigma_fx) (3) 분산효과/성향(rho, lambda)
- w_H_star가 0% 또는 100%에 가깝다면(>=95% 또는 <=5%) 'clip 가능성' 경고 1문장 포함
- 보고서/목차/섹션 형태로 쓰지 말 것. (번호/헤더/#### 금지)

답변을 생성해주세요."""
            
            self.logger.info("🤖 LLM을 사용하여 답변 생성 시작")
            
            # LLM 호출
            answer = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            answer_cleaned = answer.strip()
            
            self.logger.info(
                f"✅ 답변 생성 완료",
                {"answer_length": len(answer_cleaned)}
            )
            
            return answer_cleaned
            
        except Exception as e:
            self.logger.error(
                f"❌ 답변 생성 실패",
                {"error": str(e)},
                exc_info=True
            )
            # Fallback: 간단한 답변
            if w_H_star is not None:
                return f"최적 환헷지 비중은 {w_H_star:.2%}입니다. (비헷지 비중: {w_UH_star:.2%})"
            else:
                return "최적 환헷지 비중을 계산할 수 없습니다."
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        환헷지 전략 실행
        
        Args:
            task: 수행할 작업 설명 (사용자 요청)
            context: 추가 컨텍스트 정보
                - user_id: 사용자 ID (optional)
                - asset_col: 자산 가격 컬럼명 (optional)
                - sigma_asset_manual: 수동 설정하는 σ_Asset 값 (optional, 기본값: 0.15)
                - risk_aversion: 위험회피도 (optional, None이면 user_id에서 가져오거나 기본값 4.0)
                - alpha: 시그널 → E[R_FX] 변환 스케일 파라미터 (optional, 기본값: 0.0001)
                - rdb_days: RDB에서 가져올 최근 일수 (optional, 기본값: 252)
        
        Returns:
            계산에 사용된 입력값 및 결과를 담은 딕셔너리
        """
        self.logger.info(
            f"🔄 전략 실행 에이전트 시작",
            {"task": task, "context": context}
        )
        
        # 컨텍스트에서 파라미터 추출
        user_id = context.get("user_id") if context else None
        asset_col = context.get("asset_col") if context else None
        sigma_asset_manual = context.get("sigma_asset_manual", 0.15) if context else 0.15
        risk_aversion = context.get("risk_aversion") if context else None
        alpha = context.get("alpha", 0.0001) if context else 0.0001
        rdb_days = context.get("rdb_days", 252) if context else 252
        
        try:
            # 1) 뉴스 센티멘트 분석 → signal_score 가져오기
            self.logger.info("📰 뉴스 센티멘트 분석 시작")
            sentiment_result = run_sentiment_analysis()
            
            # 기본값: 실패 시 0.5로 고정
            signal_score = 0.5
            
            try:
                # 문자열에서 리스트 추출 시도
                if isinstance(sentiment_result, str):
                    # Python 리스트 형태인지 확인
                    match = re.search(r'\[.*?\]', sentiment_result, re.DOTALL)
                    if match:
                        data = ast.literal_eval(match.group())
                        # USD 관련 항목 찾기
                        usd_items = [item for item in data if isinstance(item, dict) and item.get("currency") == "USD"]
                        if usd_items:
                            # USD의 signal_score 가져오기 (있으면 사용, 없으면 impact 사용)
                            for item in usd_items:
                                # signal_score 필드가 있으면 사용
                                if "signal_score" in item and isinstance(item.get("signal_score"), (int, float)):
                                    signal_score = float(item["signal_score"])
                                    break
                                # signal_score가 없으면 impact를 사용 (direction에 따라 부호 조정)
                                elif "impact" in item and isinstance(item.get("impact"), (int, float)):
                                    impact = float(item.get("impact", 0.0))
                                    direction = item.get("direction", "neutral")
                                    if direction == "down":
                                        signal_score = -impact
                                    elif direction == "up":
                                        signal_score = impact
                                    else:
                                        signal_score = impact
                                    break
            except Exception as e:
                self.logger.warning(f"Warning: Failed to parse sentiment result: {e}")
                signal_score = 0.5  # 실패 시 0.5로 고정
            
            self.logger.info(f"✅ signal_score 추출 완료: {signal_score}")
            
            # 1.5) risk_aversion 가져오기 (user_id가 있으면 RDB에서, 없으면 파라미터 또는 기본값)
            if risk_aversion is None:
                risk_aversion = get_user_risk_aversion(user_id)
            else:
                self.logger.info(f"✅ risk_aversion 파라미터 사용: {risk_aversion}")
            
            # 2) RDB에서 sigma_fx, rho 계산 (CalculatorTool 메서드 직접 사용)
            self.logger.info(f"📊 RDB에서 환율 데이터 조회 시작 (최근 {rdb_days}일)")
            sigma_fx, rho = self.calculator.compute_fx_vol_and_rho_from_rdb(
                days=rdb_days,
                asset_col=asset_col,
            )
            self.logger.info(f"✅ RDB에서 환율 데이터를 가져왔습니다. sigma_fx={sigma_fx:.6f}, rho={rho:.6f}")
            
            # 3) sigma_asset은 일단 수동 입력 값 사용 (나중에 실제 자산 데이터로 대체 가능)
            sigma_asset = float(sigma_asset_manual)
            
            # 4) E[R_FX] 계산 (CalculatorTool 메서드 직접 사용)
            self.logger.info("💰 기대 환율 수익률 계산 시작")
            expected_fx_return = self.calculator.compute_expected_fx_return(
                signal_score=signal_score,
                alpha=alpha,
            )
            self.logger.info(f"✅ 기대 환율 수익률: {expected_fx_return:.6f}")
            
            # 5) 최적 환헷지 비중 w_H* 계산 (CalculatorTool 메서드 직접 사용)
            self.logger.info("⚖️ 최적 환헷지 비중 계산 시작")
            w_H_star = self.calculator.compute_optimal_hedge_weight(
                sigma_asset=sigma_asset,
                sigma_fx=sigma_fx,
                rho=rho,
                expected_fx_return=expected_fx_return,
                risk_aversion=risk_aversion,
                clip=True,
            )
            self.logger.info(f"✅ 최적 환헷지 비중: {w_H_star}")
            
            # 6) 결과 정리 및 자연어 답변 생성
            w_UH_star = None if w_H_star is None else (1.0 - w_H_star)
            
            # Tool 결과를 바탕으로 LLM을 사용하여 자연어 답변 생성
            answer = await self._generate_answer(
                task=task,
                signal_score=signal_score,
                expected_fx_return=expected_fx_return,
                sigma_asset=sigma_asset,
                sigma_fx=sigma_fx,
                rho=rho,
                risk_aversion=risk_aversion,
                alpha=alpha,
                w_H_star=w_H_star,
                w_UH_star=w_UH_star,
            )
            
            result = {
                "agent": self.name,
                "task": task,
                "answer": answer,  # 자연어 답변 추가
                "signal_score": signal_score,
                "expected_fx_return": expected_fx_return,
                "sigma_asset": sigma_asset,
                "sigma_fx": sigma_fx,
                "rho": rho,
                "risk_aversion": risk_aversion,
                "alpha": alpha,
                "w_H_star": w_H_star,
                "w_UH_star": w_UH_star,
                "status": "success"
            }
            
            self.logger.info(
                f"✅ 전략 실행 완료",
                {
                    "w_H_star": w_H_star,
                    "w_UH_star": w_UH_star,
                    "answer_length": len(answer)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"❌ 전략 실행 실패",
                {"error": str(e)},
                exc_info=True
            )
            return {
                "agent": self.name,
                "task": task,
                "answer": f"전략 실행 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "status": "error"
            }


# 하위 호환성을 위한 함수 (기존 코드에서 사용할 수 있도록)
async def run_strategy_execution(
    user_id: Optional[str] = None,
    asset_col: Optional[str] = None,
    sigma_asset_manual: float = 0.15,
    risk_aversion: Optional[float] = None,
    alpha: float = 0.0001,
    rdb_days: int = 252,
) -> dict:
    """
    환헷지 전략 실행 함수 (하위 호환성 유지)
    
    이 함수는 StrategyExecuteAgent를 사용하여 전략을 실행합니다.
    """
    agent = StrategyExecuteAgent()
    context = {
        "user_id": user_id,
        "asset_col": asset_col,
        "sigma_asset_manual": sigma_asset_manual,
        "risk_aversion": risk_aversion,
        "alpha": alpha,
        "rdb_days": rdb_days,
    }
    result = await agent.execute("환헷지 전략 실행", context)
    return result


if __name__ == "__main__":
    import asyncio
    
    async def main():
        res = await run_strategy_execution(
            user_id=None,  # 사용자 ID (예: "user123"). None이면 기본값 4.0 사용
            asset_col=None,       # RDB에 자산 가격 컬럼 있으면 이름 넣으면 됨
            sigma_asset_manual=0.15,
            risk_aversion=None,  # None이면 user_id에서 가져오거나 기본값 4.0
            alpha=0.0001,  # 일간 기준으로 적절한 값 (0.01% 수준)
            rdb_days=252,  # 최근 1년 데이터
        )
        
        print("=== 환헷지 전략 실행 결과 ===")
        for k, v in res.items():
            print(f"{k:>20}: {v}")
    
    asyncio.run(main())

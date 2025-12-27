"""
전략 실행 Agent

환헷지 전략을 수행하는 에이전트입니다.
- 뉴스 센티멘트 분석 결과 → signal_score
- RDB (eiExchangeRate 테이블) → 자산(SPY_close) 변동성(sigma_asset), 환율 변동성(sigma_fx), 상관계수(rho)
- user_info.risk_level → 위험회피도(lambda)
- 위 값들을 이용해서 최적 환헷지 비중 w_H* 계산
"""

from typing import Optional, Dict, Any
import sys
import importlib.util
from pathlib import Path
import ast
import re

# 프로젝트 루트를 Python 경로에 추가 (직접 실행 시)
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
from src.tools.calculator import CalculatorTool
from src.utils.logger import get_logger
from src.utils.llm import call_gpt
from src.utils.state import Action, Reference
# DB 저장 기능 제거: 사용자가 명시적으로 요청할 때만 업데이트하도록 변경
# from src.utils.hedge_db import ensure_hedge_tables, log_hedge_event, update_hedge_settings, get_hedge_settings
# from src.tools.rdb import user_info_get, user_info_upsert  # DB 업데이트용 - 사용하지 않음
from uuid import uuid4

# 주의: 이 agent는 헷지 비율을 계산만 하고 DB에 저장하지 않습니다.
# DB 업데이트는 사용자가 명시적으로 요청할 때 별도의 agent나 엔드포인트에서 처리해야 합니다.

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
"""


def get_user_overseas_usd(user_id: Optional[str] = None) -> Optional[float]:
    """
    RDB의 user_info 테이블에서 user_usd(해외자산 총액) 값을 가져옵니다.
    
    Args:
        user_id: 사용자 ID. None이면 None 반환
        
    Returns:
        user_usd 값 (없으면 None)
    """
    if user_id is None:
        return None
        
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
        
        conn = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_usd FROM user_info WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            
        conn.close()
        
        if result and result.get("user_usd") is not None:
            return float(result["user_usd"])
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ user_usd 조회 중 오류: {e}")
        return None


def get_user_risk_aversion(user_id: Optional[str] = None) -> float:
    """
    RDB의 user_info 테이블에서 risk_level 컬럼 값을 가져와서 risk_aversion(lambda)로 사용합니다.
    
    주의: 실제 DB 컬럼명은 risk_level입니다. 이 값이 위험회피도(lambda, risk_aversion)로 사용됩니다.

    Args:
        user_id: 사용자 ID. None이면 기본값 4.0 반환

    Returns:
        risk_level 컬럼 값 (없으면 4.0)
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

        # if not all([db_host, db_user, db_password, db_name]):
        #     logger.warning("⚠️  데이터베이스 연결 정보가 없어 기본값 4.0을 사용합니다.")
        #     return 4.0

        conn = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )

        query = """
            SELECT risk_level
            FROM user_info
            WHERE user_id = %s
            LIMIT 1
        """

        with conn.cursor() as cursor:
            # 먼저 해당 user_id가 존재하는지 확인
            check_query = "SELECT user_id, user_name, user_krw, user_usd, risk_level FROM user_info WHERE user_id = %s"
            cursor.execute(check_query, (user_id,))
            all_data = cursor.fetchone()
            
            logger.debug(f"🔍 [DEBUG] 전체 사용자 정보 조회 - user_id: {user_id}, 결과: {all_data}")
            if all_data:
                logger.debug(f"🔍 전체 데이터: {all_data}, risk_level: {all_data.get('risk_level')}, 타입: {type(all_data.get('risk_level'))}")
            
            # risk_level만 조회
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            
            # 디버깅: 쿼리 결과 확인
            logger.debug(f"🔍 [DEBUG] risk_level만 조회한 결과: {result}")
            if result:
                logger.debug(f"🔍 result type: {type(result)}, keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
                logger.debug(f"🔍 result.get('risk_level'): {result.get('risk_level')}, type: {type(result.get('risk_level'))}")
            else:
                logger.warning(f"⚠️ 쿼리 결과가 None입니다. user_id '{user_id}'가 DB에 없을 수 있습니다.")
                # 모든 user_id 목록 확인
                cursor.execute("SELECT user_id FROM user_info LIMIT 10")
                all_users = cursor.fetchall()
                user_ids = [u.get('user_id') for u in all_users] if all_users else []
                logger.debug(f"🔍 DB에 있는 user_id 목록 (최대 10개): {user_ids}")

        conn.close()

        if result:
            risk_level_value = result.get("risk_level")
            logger.debug(f"🔍 risk_level_value: {risk_level_value}, type: {type(risk_level_value)}")
            
            if risk_level_value is not None:
                try:
                    # DB의 risk_level 컬럼 값을 risk_aversion(lambda)로 사용
                    # 문자열로 저장된 경우(예: "0.2d") 처리
                    if isinstance(risk_level_value, str):
                        # "0.2d" 같은 경우 숫자 부분만 추출
                        import re
                        match = re.search(r'[\d.]+', risk_level_value)
                        if match:
                            risk_aversion = float(match.group())
                        else:
                            risk_aversion = float(risk_level_value)
                    else:
                        risk_aversion = float(risk_level_value)
                    
                    logger.info(f"✅ 사용자 {user_id}의 risk_level(DB) → risk_aversion(lambda): {risk_aversion}")
                    return risk_aversion
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️  risk_level 값을 숫자로 변환 실패: {risk_level_value}, 오류: {e}")
            else:
                logger.warning(f"⚠️  사용자 {user_id}의 risk_level(DB) 값이 None입니다.")

        logger.warning(f"⚠️  사용자 {user_id}의 risk_level(DB) 정보가 없어 기본값 4.0을 사용합니다.")
        return 4.0

    except ImportError:
        logger.warning("⚠️  pymysql이 설치되지 않아 기본값 4.0을 사용합니다.")
        return 4.0
    except Exception as e:
        logger.warning(f"⚠️  RDB에서 risk_level(DB)을 가져오는 중 오류 발생: {e}")
        logger.warning("   기본값 4.0을 사용합니다.")
        return 4.0


class StrategyExecuteAgent(BaseAgent):
    """환헷지 전략을 수행하는 에이전트"""

    def __init__(self):
        super().__init__(
            name="strategy_execute",
            description="환헷지 전략을 수행하는 전문 에이전트",
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
        계산 결과를 바탕으로 LLM을 사용하여 자연어 답변 생성
        (형식 강제: bullet 4~5줄, 숫자/부호/단위 변경 금지)
        """
        try:
            if w_H_star is None or w_UH_star is None:
                return "⚠️ 최적 환헷지 비중을 계산할 수 없습니다. 입력 파라미터를 확인해주세요."

            # ---- Formatting helpers ----
            def fmt_pct(x: float, nd: int = 1) -> str:
                return f"{x * 100:.{nd}f}%"

            def fmt_ret(x: float) -> str:
                return f"{x:.6f} ({x * 100:.3f}%)"

            def fmt_vol(x: float) -> str:
                return f"{x * 100:.2f}%"
            
            def fmt_signal_score(x: float) -> str:
                """signal_score 포맷팅 (소수점 4자리)"""
                return f"{x:.4f}"

            def fx_view(er: float) -> str:
                if er > 0:
                    return "USD 강세(USD stronger) 신호"
                if er < 0:
                    return "USD 약세(USD weaker) 신호"
                return "중립(Neutral) 신호"

            fx_outlook = fx_view(expected_fx_return)
            hedge_pressure = (
                "헷지 비중↑(hedge up) 압력" if expected_fx_return < 0 else
                "헷지 비중↓(hedge down) 압력" if expected_fx_return > 0 else
                "중립(Neutral) 압력"
            )

            formatted = {
                "w_H_star_pct": fmt_pct(w_H_star),
                "w_UH_star_pct": fmt_pct(w_UH_star),
                "signal_score_fmt": fmt_signal_score(signal_score),
                "expected_fx_return_fmt": fmt_ret(expected_fx_return),
                "sigma_asset_fmt": fmt_vol(sigma_asset),
                "sigma_fx_fmt": fmt_vol(sigma_fx), 
                "rho_fmt": f"{rho:.3f}",
                "lambda_fmt": f"{risk_aversion:.2f}",
                "fx_outlook": fx_outlook,
                "hedge_pressure": hedge_pressure,
            }

            # clip 경고 조건
            clip_warn = (w_H_star <= 0.05) or (w_H_star >= 0.95)

            system_prompt = """당신은 환헷지(FX hedging) 전략 전문가입니다.

## 역할
사용자에게 계산된 환헷지 전략 결과를 명확하고 이해하기 쉽게 설명합니다.

## 출력 형식 (반드시 준수)

### Part 1: 요약 설명 (3~4문장의 자연스러운 단락)
- 첫 문장: 결론 (권장 환헷지/환노출 비율)
- 둘째 문장: 환율 신호 해석 (signal_score, expected_fx_return, 방향성)
- 셋째 문장: 리스크 요인들(변동성, 상관계수, 위험회피도)이 비율에 미친 영향
- (조건부) clip 경고가 있으면 마지막에 경고 문장 추가

### Part 2: 핵심 지표 테이블 (요약 설명 아래에 고정 형식으로 출력)

## 필수 규칙
- 제공된 숫자/부호/단위를 절대 변경하지 마세요
- "유리/불리"처럼 단정짓지 말고, "압력" 또는 "신호"로 표현하세요
- expected_fx_return 해석:
  - 양수 → USD 강세 신호 → 헷지 비중 감소 압력
  - 음수 → USD 약세 신호 → 헷지 비중 증가 압력
  - 0 → 중립
"""

            user_prompt = f"""## 계산 결과
- 권장 환헷지 비율 (w_H*): {formatted["w_H_star_pct"]}
- 권장 환노출 비율 (w_UH*): {formatted["w_UH_star_pct"]}
- signal_score: {formatted["signal_score_fmt"]}
- expected_fx_return: {formatted["expected_fx_return_fmt"]}
- σ_asset (자산 변동성): {formatted["sigma_asset_fmt"]}
- σ_fx (환율 변동성): {formatted["sigma_fx_fmt"]}
- ρ (상관계수): {formatted["rho_fmt"]}
- λ (위험회피도): {formatted["lambda_fmt"]}

## 해석 힌트
- 환율 방향: {formatted["fx_outlook"]}
- 헷지 압력: {formatted["hedge_pressure"]}

위 정보를 바탕으로:
1. 자연스러운 3~4문장의 요약 설명을 작성하세요.
2. 마지막에 아래 형식의 핵심 지표 테이블을 그대로 출력하세요:

📊 핵심 지표
• 권장 헷지 비율: {formatted["w_H_star_pct"]} / 환노출 비율: {formatted["w_UH_star_pct"]}
• 환율 신호: {formatted["signal_score_fmt"]} → {formatted["fx_outlook"]}
• 변동성: σ_asset={formatted["sigma_asset_fmt"]}, σ_fx={formatted["sigma_fx_fmt"]}, ρ={formatted["rho_fmt"]}
• 위험회피도: λ={formatted["lambda_fmt"]}
"""

            if clip_warn:
                user_prompt += "\n⚠️ 경고: 계산된 비율이 경계값(5% 또는 95%)에 가까워 클리핑된 결과일 수 있습니다."

            self.logger.info("🤖 LLM을 사용하여 답변 생성 시작")

            answer = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            answer_cleaned = (answer or "").strip()
            self.logger.info("✅ 답변 생성 완료", {"answer_length": len(answer_cleaned)})
            return answer_cleaned

        except Exception as e:
            self.logger.error("❌ 답변 생성 실패", {"error": str(e)}, exc_info=True)
            fallback = f"""현재 시장 상황을 분석한 결과, 권장 환헷지 비율은 {w_H_star:.1%}이며 환노출 비율은 {w_UH_star:.1%}입니다.

📊 핵심 지표
• 권장 헷지 비율: {w_H_star:.1%} / 환노출 비율: {w_UH_star:.1%}
• 환율 신호: {signal_score:.4f}
• 변동성: σ_asset={sigma_asset*100:.2f}%, σ_fx={sigma_fx*100:.2f}%, ρ={rho:.3f}
• 위험회피도: λ={risk_aversion:.2f}

(LLM 답변 생성 중 오류: {str(e)})"""
            return fallback

    async def _confirm_hedge_ratio_with_user(
        self,
        calculated_ratio: float,
        user_message: Optional[str] = None,
    ) -> Optional[float]:
        """
        사용자와 대화를 통해 헷지 비율을 확정합니다.
        
        Args:
            calculated_ratio: 계산된 헷지 비율
            user_message: 사용자의 응답 메시지 (None이면 None 반환)
            
        Returns:
            확정된 헷지 비율 (None이면 아직 확정되지 않음)
        """
        if user_message is None:
            # 사용자 응답이 없으면 None 반환 (확정되지 않음)
            return None
        
        # 사용자 응답 분석
        system_prompt = """당신은 사용자의 응답을 분석하여 헷지 비율 확정 여부를 판단하는 전문가입니다.

사용자 응답 분석 규칙:
1. '예', '확인', '적용', '좋아', 'OK' 등 긍정적 응답 → 계산된 비율 사용
2. 숫자나 비율이 명시된 경우 (예: 0.3, 30%, 25% 등) → 해당 비율 사용
3. '아니오', '취소', '안 할래' 등 부정적 응답 → None 반환
4. 모호한 경우 → 계산된 비율 사용

반드시 다음 JSON 형식으로만 응답하세요:
{
  "confirmed": true/false,
  "ratio": <확정된 비율 (0.0~1.0) 또는 null>,
  "reasoning": "<판단 근거>"
}"""

        user_prompt = f"""계산된 헷지 비율: {calculated_ratio:.1%} ({calculated_ratio:.4f})

사용자 응답: {user_message}

위 사용자 응답을 분석하여 헷지 비율 확정 여부와 값을 판단하세요."""

        try:
            response = await call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            
            import json
            result = json.loads(response)
            
            if result.get("confirmed", False) and result.get("ratio") is not None:
                confirmed_ratio = float(result["ratio"])
                # 비율 범위 검증
                if 0.0 <= confirmed_ratio <= 1.0:
                    self.logger.info(
                        f"✅ 헷지 비율 확정됨",
                        {
                            "calculated_ratio": calculated_ratio,
                            "confirmed_ratio": confirmed_ratio,
                            "reasoning": result.get("reasoning", ""),
                        }
                    )
                    return confirmed_ratio
                else:
                    self.logger.warning(
                        f"⚠️ 확정된 비율이 범위를 벗어남, 계산된 비율 사용",
                        {"confirmed_ratio": confirmed_ratio, "calculated_ratio": calculated_ratio}
                    )
                    return calculated_ratio
            elif result.get("confirmed", False):
                # 확정되었지만 비율이 없으면 계산된 비율 사용
                self.logger.info(
                    f"✅ 헷지 비율 확정됨 (계산된 비율 사용)",
                    {"calculated_ratio": calculated_ratio, "reasoning": result.get("reasoning", "")}
                )
                return calculated_ratio
            else:
                # 확정되지 않음
                self.logger.info(
                    f"ℹ️ 헷지 비율 확정되지 않음",
                    {"reasoning": result.get("reasoning", "")}
                )
                return None
                
        except Exception as e:
            self.logger.error(
                f"❌ 헷지 비율 확정 분석 실패",
                {"error": str(e), "user_message": user_message},
                exc_info=True
            )
            # 오류 시 계산된 비율 사용
            return calculated_ratio


    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """환헷지 전략 실행"""
        self.logger.info("🔄 전략 실행 에이전트 시작", {"task": task, "context": context})

        user_id = context.get("user_id") if context else None
        asset_col = context.get("asset_col") if context else None
        risk_aversion = context.get("risk_aversion") if context else None
        alpha = context.get("alpha", 0.0001) if context else 0.0001
        rdb_days = context.get("rdb_days", 252) if context else 252

        try:
            # 1) 뉴스 센티멘트 분석 → signal_score
            self.logger.info("📰 뉴스 센티멘트 분석 시작")
            sentiment_result = run_sentiment_analysis()

            # signal_score 초기값 설정 (기본값: 0.0, 중립)
            signal_score = 0.0
            
            try:
                if isinstance(sentiment_result, str):
                    match = re.search(r"\[.*?\]", sentiment_result, re.DOTALL)
                    if match:
                        data = ast.literal_eval(match.group())
                        usd_items = [
                            item for item in data
                            if isinstance(item, dict) and item.get("currency") == "USD"
                        ]
                        if usd_items:
                            for item in usd_items:
                                # impact와 direction을 사용하여 signal_score 계산
                                if "impact" in item and isinstance(item.get("impact"), (int, float)):
                                    impact = float(item.get("impact", 0.0))
                                    direction = item.get("direction", "neutral")
                                    
                                    # direction에 따라 signal_score 계산
                                    # down: USD 약세 → 음수, up: USD 강세 → 양수, neutral: 중립 → 0
                                    if direction == "down":
                                        signal_score = -impact
                                    elif direction == "up":
                                        signal_score = impact
                                    elif direction == "neutral":
                                        signal_score = 0.0
                                    else:
                                        # 알 수 없는 direction인 경우 impact를 그대로 사용
                                        signal_score = impact
                                    
                                    self.logger.info(
                                        "✅ signal_score 계산 완료",
                                        {
                                            "impact": impact,
                                            "direction": direction,
                                            "signal_score": signal_score
                                        }
                                    )
                                    break
                        else:
                            self.logger.warning("⚠️ USD 관련 항목을 찾을 수 없어 기본값(0.0)을 사용합니다.")
                    else:
                        self.logger.warning("⚠️ 센티멘트 결과에서 데이터를 파싱할 수 없어 기본값(0.0)을 사용합니다.")
                else:
                    self.logger.warning("⚠️ 센티멘트 결과가 문자열이 아니어 기본값(0.0)을 사용합니다.")
            except Exception as e:
                self.logger.warning(
                    f"⚠️ 센티멘트 결과 파싱 실패, 기본값(0.0) 사용",
                    {"error": str(e), "sentiment_result_type": type(sentiment_result).__name__}
                )
                signal_score = 0.0

            self.logger.info(f"✅ signal_score 최종값: {signal_score:.4f}")

            # 1.5) risk_aversion(lambda) - DB의 risk_level 컬럼에서 가져옴
            if risk_aversion is None:
                risk_aversion = get_user_risk_aversion(user_id)  # DB의 risk_level → risk_aversion
            else:
                self.logger.info(f"✅ risk_aversion 파라미터 사용: {risk_aversion} (DB의 risk_level과 동일)")

            # 2) RDB에서 sigma_asset, sigma_fx, rho
            if asset_col is None:
                asset_col = "SPY_close"

            self.logger.info(f"📊 RDB에서 자산({asset_col})/환율 데이터 조회 시작 (최근 {rdb_days}일)")
            sigma_asset, sigma_fx, rho = self.calculator.compute_sigmas_and_rho_from_rdb(
                days=rdb_days,
                asset_col=asset_col,
            )
            self.logger.info(
                "✅ RDB에서 자산/환율 데이터를 가져왔습니다.",
                {"sigma_asset": sigma_asset, "sigma_fx": sigma_fx, "rho": rho},
            )

            # 3) E[R_FX]
            self.logger.info("💰 기대 환율 수익률 계산 시작")
            expected_fx_return = self.calculator.compute_expected_fx_return(
                signal_score=signal_score,
                alpha=alpha,
            )
            self.logger.info(f"✅ 기대 환율 수익률: {expected_fx_return:.6f}")

            # 4) w_H*
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

            w_UH_star = None if w_H_star is None else (1.0 - w_H_star)

            # 5) LLM 답변
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

            # 6) 헷지 비율 계산 완료 (DB 저장 제거: 사용자가 명시적으로 요청할 때만 업데이트)
            # 이 agent는 헷지 비율을 계산만 하고, DB 저장은 하지 않습니다.
            # DB 업데이트는 사용자가 명시적으로 요청할 때 별도로 처리됩니다.
            confirmed_ratio = None
            update_result = None
            run_id = str(uuid4())  # 이번 실행의 고유 ID
            
            self.logger.info(
                "✅ 헷지 비율 계산 완료 (DB 저장 없음)",
                {
                    "user_id": user_id,
                    "w_H_star": w_H_star,
                    "run_id": run_id,
                }
            )

            # Action 생성 (다른 agent가 이 결과를 찾을 수 있도록)
            action = Action(
                type="calculate",
                tool="strategy_execute",
                description=f"환헷지 전략 실행: 최적 헷지 비율 계산 완료",
                input={
                    "user_id": user_id,
                    "signal_score": signal_score,
                    "risk_aversion": risk_aversion,
                    "alpha": alpha
                },
                output={
                    "w_H_star": w_H_star,
                    "w_UH_star": w_UH_star,
                    "signal_score": signal_score,
                    "expected_fx_return": expected_fx_return,
                    "sigma_asset": sigma_asset,
                    "sigma_fx": sigma_fx,
                    "rho": rho,
                    "risk_aversion": risk_aversion,
                    "run_id": run_id
                }
            )

            result = {
                "agent": self.name,
                "task": task,
                "answer": answer,
                "signal_score": signal_score,
                "expected_fx_return": expected_fx_return,
                "sigma_asset": sigma_asset,
                "sigma_fx": sigma_fx,
                "rho": rho,
                "risk_aversion": risk_aversion,
                "alpha": alpha,
                "w_H_star": w_H_star,
                "w_UH_star": w_UH_star,
                "confirmed_hedge_ratio": confirmed_ratio,
                "hedge_update": None,  # DB 저장 제거됨
                "run_id": run_id,
                "status": "success",
                "action": [action],  # Action 추가: 다른 agent가 찾을 수 있도록
                "reference": []  # Reference는 없음
            }

            self.logger.info(
                "✅ 전략 실행 완료",
                {
                    "w_H_star": w_H_star,
                    "w_UH_star": w_UH_star,
                    "answer_length": len(answer)
                }
            )
            return result

        except Exception as e:
            self.logger.error("❌ 전략 실행 실패", {"error": str(e)}, exc_info=True)
            return {
                "agent": self.name,
                "task": task,
                "answer": f"전략 실행 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "status": "error",
            }


async def run_strategy_execution(
    user_id: Optional[str] = None,
    asset_col: Optional[str] = None,
    risk_aversion: Optional[float] = None,
    alpha: float = 0.0001,
    rdb_days: int = 252,
) -> dict:
    """환헷지 전략 실행 함수 (하위 호환성 유지)"""
    agent = StrategyExecuteAgent()
    context = {
        "user_id": user_id,
        "asset_col": asset_col,
        "risk_aversion": risk_aversion,
        "alpha": alpha,
        "rdb_days": rdb_days,
    }
    return await agent.execute("환헷지 전략 실행", context)


if __name__ == "__main__":
    import asyncio
    import argparse

    print("RUNNING FILE:", Path(__file__).resolve())

    parser = argparse.ArgumentParser(description="환헷지 전략 실행 테스트")
    parser.add_argument("--user_id", type=str, default="M000831", help="사용자 ID (기본값: M000831)")
    parser.add_argument("--asset_col", type=str, default=None, help="자산 컬럼명 (기본값: SPY_close)")
    parser.add_argument("--risk_aversion", type=float, default=None, help="위험회피도 (기본값: DB에서 가져옴)")
    parser.add_argument("--alpha", type=float, default=0.0001, help="스케일링 파라미터 (기본값: 0.0001)")
    parser.add_argument("--rdb_days", type=int, default=252, help="RDB에서 가져올 일수 (기본값: 252)")
    parser.add_argument("--no_auto_update", dest="auto_update", action="store_false", default=True, help="자동 업데이트 비활성화 (기본값: 자동 업데이트 활성화)")

    args = parser.parse_args()

    async def main():
        # 테스트 모드에서는 기본적으로 auto_update=True (확정 요청 없이 바로 업데이트)
        auto_update = getattr(args, 'auto_update', True)
        
        context = {
            "user_id": args.user_id,
            "asset_col": args.asset_col,
            "risk_aversion": args.risk_aversion,
            "alpha": args.alpha,
            "rdb_days": args.rdb_days,
            "auto_update": auto_update,
        }
        
        agent = StrategyExecuteAgent()
        res = await agent.execute("환헷지 전략 실행", context)

        print("\n=== 환헷지 전략 실행 결과 ===")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"{k:>25}: {v:.6f}")
            elif isinstance(v, dict):
                print(f"{k:>25}: {v}")
            else:
                print(f"{k:>25}: {v}")

    asyncio.run(main())

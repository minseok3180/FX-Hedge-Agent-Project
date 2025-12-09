"""User Information 에이전트

사용자 프로필/자산 정보를 조회·입력·수정하는 전용 에이전트.

- MariaDB `user_info` 테이블을 사용
- 툴: `user_info_get`, `user_info_upsert`
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.utils.agents import BaseAgent
from src.utils.logger import get_logger
from src.utils.llm import call_gpt
from src.tools.rdb import user_info_get, user_info_upsert
from src.prompts.user_information_instruction import USER_INFORMATION_INSTRUCTION


logger = get_logger(__name__)


class UserInformationAgent(BaseAgent):
    """사용자 프로필 및 자산 정보를 관리하는 에이전트."""

    def __init__(self) -> None:
        super().__init__(
            name="user_information",
            description=(
                "사용자 프로필(이름, 나이, 성별, 총재산, 해외재산, 투자성향)을 "
                "MariaDB user_info 테이블에 저장/수정하고, 필요한 에이전트에게 제공하는 에이전트"
            ),
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """사용자 정보 조회 또는 수정/등록을 수행한다."""
        context = context or {}
        state = context.get("state") or {}
        user_id: Optional[str] = (
            context.get("user_id")
            or state.get("user_id")
            or state.get("current_context", {}).get("user_id")
        )

        if not user_id:
            msg = "user_id가 제공되지 않아 사용자 정보를 조회/수정할 수 없습니다."
            logger.error(
                "❌ UserInformationAgent.execute - user_id 누락",
                {"task": task, "context_keys": list(context.keys())},
            )
            return {
                "agent": self.name,
                "status": "error",
                "mode": None,
                "answer": msg,
                "user_info": None,
                "reasoning": "context/state에 user_id가 없어 작업을 수행할 수 없음",
            }

        try:
            # 1) LLM에 어떤 작업을 할지 판단 요청 (조회 vs 수정/등록, 어떤 필드인지 등)
            plan = await self._analyze_request(task=task, user_id=user_id, context=context)

            mode = plan.get("mode") or plan.get("action")  # "get" | "upsert"
            reasoning = plan.get("reasoning", "")
            fields: Dict[str, Any] = plan.get("fields", {}) or {}

            if mode not in ("get", "upsert"):
                logger.warning(
                    "⚠️ UserInformationAgent - mode가 명확하지 않아 기본값(get) 사용",
                    {"mode": mode, "plan": plan},
                )
                mode = "get"

            # 2) 실제 DB 작업 수행
            if mode == "get":
                result = await user_info_get(user_id=user_id)
                if result.get("found"):
                    user_info_data = result.get("user_info")
                    answer = self._format_user_info_answer(user_info_data, prefix="현재 등록된 사용자 정보입니다.")
                else:
                    user_info_data = None
                    answer = (
                        "현재 이 user_id에 해당하는 사용자 정보가 등록되어 있지 않습니다. "
                        "필요하다면 이름, 나이, 성별, 총재산, 해외재산, 투자성향을 알려주면 새로 등록할 수 있습니다."
                    )

                return {
                    "agent": self.name,
                    "status": "success",
                    "mode": "get",
                    "answer": answer,
                    "user_info": user_info_data,
                    "reasoning": reasoning,
                }

            # mode == "upsert"
            # 부족한 필드는 기존 DB 값으로 보완 (있다면)
            existing = await user_info_get(user_id=user_id)
            existing_info: Dict[str, Any] = existing.get("user_info") or {}

            merged_fields: Dict[str, Any] = {
                "user_id": user_id,
                "name": fields.get("name", existing_info.get("name", "")),
                "age": fields.get("age", existing_info.get("age", 0)),
                "gender": fields.get("gender", existing_info.get("gender", "")),
                "total_assets": fields.get("total_assets", existing_info.get("total_assets", 0.0)),
                "overseas_assets": fields.get(
                    "overseas_assets",
                    existing_info.get("overseas_assets", 0.0),
                ),
                "risk_profile": fields.get("risk_profile", existing_info.get("risk_profile", "")),
            }

            upsert_result = await user_info_upsert(**merged_fields)
            saved_info = upsert_result.get("user_info")
            operation = upsert_result.get("operation")

            if operation == "insert":
                prefix = "새로운 사용자 정보를 등록했습니다."
            else:
                prefix = "기존 사용자 정보를 업데이트했습니다."

            answer = self._format_user_info_answer(saved_info, prefix=prefix)

            return {
                "agent": self.name,
                "status": "success",
                "mode": "upsert",
                "answer": answer,
                "user_info": saved_info,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error(
                "❌ UserInformationAgent.execute 실패",
                {"error": str(e), "task": task},
                exc_info=True,
            )
            return {
                "agent": self.name,
                "status": "error",
                "mode": None,
                "answer": f"사용자 정보 처리 중 오류가 발생했습니다: {e}",
                "user_info": None,
                "reasoning": "",
            }

    async def _analyze_request(
        self,
        task: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LLM으로부터 '조회' vs '수정/등록' 및 필드 추출 계획을 받아온다."""
        context = context or {}

        system_prompt = f"""{USER_INFORMATION_INSTRUCTION}

## 응답 형식
반드시 다음 JSON 형식으로만 응답하세요 (설명 텍스트 금지):
{{
  "mode": "get" | "upsert",
  "fields": {{
    "name": "<이름 또는 null>",
    "age": <나이 또는 null>,
    "gender": "<성별 또는 null>",
    "total_assets": <총재산 또는 null>,
    "overseas_assets": <해외재산 또는 null>,
    "risk_profile": "<투자성향 또는 null>"
  }},
  "reasoning": "<당신의 판단 근거>"
}}

필드 값이 요청에서 전혀 언급되지 않았다면 null 또는 생략해도 됩니다.
"""

        user_context_str = ""
        if context:
            try:
                user_context_str = json.dumps(
                    {"user_id": user_id, "context_keys": list(context.keys())},
                    ensure_ascii=False,
                )
            except Exception:
                user_context_str = f'{{"user_id": "{user_id}"}}'

        user_prompt = f"""사용자(또는 다른 에이전트)의 요청:
{task}

현재 user_id: {user_id}
간단한 컨텍스트 정보: {user_context_str}

이 요청이 사용자 정보 "조회"인지 "수정/등록(upsert)"인지 판단하고,
필요하다면 어떤 필드(name, age, gender, total_assets, overseas_assets, risk_profile)를
어떤 값으로 저장/수정해야 하는지 위 응답 형식에 맞추어 JSON으로만 반환하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await call_gpt(
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(
                "⚠️ UserInformationAgent._analyze_request - JSON 파싱 실패, 기본값 사용",
                {"response_preview": response[:200]},
            )
            plan = {
                "mode": "get",
                "fields": {},
                "reasoning": "LLM 응답 파싱 실패로 기본 조회 모드를 사용함",
            }

        if "fields" not in plan or not isinstance(plan["fields"], dict):
            plan["fields"] = {}

        return plan

    def _format_user_info_answer(
        self,
        user_info: Optional[Dict[str, Any]],
        prefix: str = "",
    ) -> str:
        """user_info 딕셔너리를 사람이 읽기 좋은 한국어 문장으로 변환."""
        if not user_info:
            return prefix or "사용자 정보가 없습니다."

        name = user_info.get("name") or "이름 미등록"
        age = user_info.get("age")
        gender = user_info.get("gender") or "성별 미등록"
        total_assets = user_info.get("total_assets")
        overseas_assets = user_info.get("overseas_assets")
        risk_profile = user_info.get("risk_profile") or "투자성향 미등록"

        def fmt_money(v: Any) -> str:
            try:
                return f"{float(v):,.0f}원"
            except Exception:
                return "정보 없음"

        parts = []
        if prefix:
            parts.append(prefix)

        parts.append(f"이름: {name}")
        if age is not None:
            parts.append(f"나이: {age}세")
        parts.append(f"성별: {gender}")
        if total_assets is not None:
            parts.append(f"총 재산: {fmt_money(total_assets)}")
        if overseas_assets is not None:
            parts.append(f"해외 재산: {fmt_money(overseas_assets)}")
        parts.append(f"투자 성향: {risk_profile}")

        return " / ".join(parts)



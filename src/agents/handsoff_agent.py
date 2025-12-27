"""Hands-off 에이전트 - 직접 답변할지 Supervisor에게 넘길지 결정"""
from typing import Dict, Any, Optional
from src.utils.agents import BaseAgent


HANDSOFF_SYSTEM_PROMPT = """당신은 에이전트로서 사용자 질문에 직접 답변할지, Supervisor에게 넘길지 결정하는 전문가입니다.

## 판단 기준

### 직접 답변 (forward)
다음 조건을 모두 만족할 때:
1. **충분한 정보**: 답변에 필요한 모든 정보가 수집되었을 때
2. **명확한 답변 가능**: 현재 정보로 명확하고 정확한 답변을 제공할 수 있을 때
3. **단일 도메인**: 하나의 도메인 내에서 해결 가능한 질문일 때

### Supervisor에게 넘기기 (handsoff)
다음 조건 중 하나라도 만족할 때:
1. **정보 부족**: 추가 정보 수집이 필요할 때
2. **다른 에이전트 필요**: 다른 에이전트의 도움이 필요할 때
3. **복합 질문**: 여러 도메인에 걸친 복합적인 질문일 때
4. **불확실성**: 정보가 불확실하거나 검증이 필요할 때

## 응답 형식
다음 JSON 형식으로 응답하세요:
{
    "decision": "forward" 또는 "handsoff",
    "reasoning": "판단 근거",
    "answer": "직접 답변 내용 (decision이 forward일 때만)",
    "handoff_reason": "Supervisor에게 넘기는 이유 (decision이 handsoff일 때만)",
    "required_agents": ["필요한 에이전트 목록 (decision이 handsoff일 때만)"]
}
"""


class HandsOffAgent(BaseAgent):
    """직접 답변할지 Supervisor에게 넘길지 결정하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="handsoff",
            description="직접 답변할지 Supervisor에게 넘길지 결정하는 에이전트"
        )
    
    async def decide(self, user_query: str, collected_data: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        직접 답변할지 Supervisor에게 넘길지 결정
        
        Args:
            user_query: 사용자 질문
            collected_data: 수집된 데이터
            context: 추가 컨텍스트
            
        Returns:
            결정 결과
        """
        self.logger.info(
            f"🤔 Hands-off 결정 시작",
            {
                "query": user_query[:100],
                "has_collected_data": collected_data is not None,
                "has_context": context is not None
            }
        )
        
        try:
            messages = [
                {"role": "system", "content": HANDSOFF_SYSTEM_PROMPT}
            ]
            
            # 수집된 데이터 및 컨텍스트 포함
            user_content = f"사용자 질문: {user_query}\n\n"
            
            try:
                if collected_data:
                    formatted_data = self._format_data(collected_data)
                    user_content += f"수집된 데이터:\n{formatted_data}\n\n"
            except Exception as e:
                self.logger.warning(
                    f"⚠️ 수집된 데이터 포맷팅 실패, 생략",
                    {"error": str(e), "error_type": type(e).__name__}
                )
            
            try:
                if context:
                    # context에서 state 같은 큰 객체는 제외하고 필수 정보만 포함
                    context_summary = {
                        "user_id": context.get("user_id"),
                        "date": context.get("date"),
                        "collected_data_keys": list(context.get("collected_data", {}).keys()) if isinstance(context.get("collected_data"), dict) else None,
                    }
                    formatted_context = self._format_context(context_summary)
                    user_content += f"추가 컨텍스트:\n{formatted_context}\n\n"
            except Exception as e:
                self.logger.warning(
                    f"⚠️ 컨텍스트 포맷팅 실패, 생략",
                    {"error": str(e), "error_type": type(e).__name__}
                )
            
            user_content += "위 정보를 바탕으로 직접 답변할지 Supervisor에게 넘길지 결정하세요."
            
            messages.append({"role": "user", "content": user_content})
            
            # LLM 호출
            response = await self._call_llm(messages, temperature=0.3)
            
            # JSON 파싱
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 기본값 (handsoff)
                result = {
                    "decision": "handsoff",
                    "reasoning": "응답 파싱 실패",
                    "answer": "",
                    "handoff_reason": "응답 형식 오류",
                    "required_agents": []
                }
            
            self.logger.info(
                f"✅ Hands-off 결정 완료",
                {
                    "decision": result.get("decision", "handsoff"),
                    "reasoning": result.get("reasoning", "")[:100]
                }
            )
            
            return {
                "agent": self.name,
                "decision": result.get("decision", "handsoff"),
                "reasoning": result.get("reasoning", ""),
                "answer": result.get("answer", ""),
                "handoff_reason": result.get("handoff_reason", ""),
                "required_agents": result.get("required_agents", []),
                "status": "success"
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            self.logger.error(
                f"❌ Hands-off 결정 실패",
                {
                    "error": error_msg,
                    "error_type": error_type,
                    "user_query": user_query[:100] if user_query else None,
                    "has_collected_data": collected_data is not None,
                    "has_context": context is not None,
                    "collected_data_keys": list(collected_data.keys()) if isinstance(collected_data, dict) else None,
                    "context_keys": list(context.keys()) if isinstance(context, dict) else None,
                },
                exc_info=True
            )
            return {
                "agent": self.name,
                "decision": "handsoff",
                "reasoning": "",
                "answer": "",
                "handoff_reason": f"에러 발생 ({error_type}): {error_msg}",
                "required_agents": [],
                "error": error_msg,
                "error_type": error_type,
                "status": "error"
            }
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """데이터를 문자열로 포맷팅"""
        import json
        try:
            # JSON 직렬화 가능한 객체만 추출
            serializable_data = self._make_serializable(data)
            return json.dumps(serializable_data, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(
                f"⚠️ 데이터 포맷팅 실패, 간단한 요약으로 대체",
                {"error": str(e)}
            )
            # 실패 시 간단한 요약만 반환
            return f"수집된 데이터 키: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 문자열로 포맷팅"""
        import json
        try:
            # JSON 직렬화 가능한 객체만 추출
            serializable_context = self._make_serializable(context)
            return json.dumps(serializable_context, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(
                f"⚠️ 컨텍스트 포맷팅 실패, 간단한 요약으로 대체",
                {"error": str(e)}
            )
            # 실패 시 간단한 요약만 반환
            return f"컨텍스트 키: {list(context.keys()) if isinstance(context, dict) else 'N/A'}"
    
    def _make_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        import json
        from datetime import datetime, date
        from decimal import Decimal
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, type):
            # type 객체는 문자열로 변환
            return str(obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, dict):
            # 객체인 경우 __dict__만 추출 (dict는 이미 위에서 처리됨)
            try:
                return self._make_serializable(obj.__dict__)
            except Exception:
                return str(obj)
        else:
            # 그 외의 경우 문자열로 변환
            return str(obj)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        BaseAgent 인터페이스 구현
        """
        collected_data = context.get("collected_data", {}) if context else {}
        return await self.decide(task, collected_data, context)


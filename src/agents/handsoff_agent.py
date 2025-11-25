"""Hands-off 에이전트 - 직접 답변할지 Supervisor에게 넘길지 결정"""
from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent


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
            
            if collected_data:
                user_content += f"수집된 데이터:\n{self._format_data(collected_data)}\n\n"
            
            if context:
                user_content += f"추가 컨텍스트:\n{self._format_context(context)}\n\n"
            
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
            self.logger.error(
                f"❌ Hands-off 결정 실패",
                {"error": str(e)},
                exc_info=True
            )
            return {
                "agent": self.name,
                "decision": "handsoff",
                "reasoning": "",
                "answer": "",
                "handoff_reason": f"에러 발생: {str(e)}",
                "required_agents": [],
                "error": str(e),
                "status": "error"
            }
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """데이터를 문자열로 포맷팅"""
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 문자열로 포맷팅"""
        import json
        return json.dumps(context, ensure_ascii=False, indent=2)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        BaseAgent 인터페이스 구현
        """
        collected_data = context.get("collected_data", {}) if context else {}
        return await self.decide(task, collected_data, context)


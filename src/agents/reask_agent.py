"""ReAsk 에이전트 - 답변에 필요한 정보가 부족할 경우 재질문"""
from typing import Dict, Any, Optional, List
from src.utils.agents import BaseAgent
from src.utils.logger import get_logger


REASK_SYSTEM_PROMPT = """당신은 사용자 질문에 답변하기 위해 필요한 정보를 확인하는 전문가입니다.

## 역할
사용자의 질문을 분석하여 답변에 필요한 정보가 충분한지 판단하고, 부족한 정보가 있다면 사용자에게 재질문합니다.

## 판단 기준
1. **필수 정보 확인**: 답변에 필요한 핵심 정보가 모두 제공되었는지 확인
2. **모호함 해소**: 질문이 모호하거나 여러 해석이 가능한 경우 명확화
3. **컨텍스트 부족**: 과거 대화나 추가 컨텍스트가 필요한 경우

## 재질문 규칙
- 한 번에 하나의 핵심 정보만 물어봅니다
- 친절하고 명확한 질문을 합니다
- 왜 그 정보가 필요한지 간단히 설명합니다
- 예시를 제공하여 사용자가 쉽게 답변할 수 있도록 합니다

## 응답 형식
다음 JSON 형식으로 응답하세요:
{
    "needs_clarification": true/false,
    "missing_info": ["부족한 정보 목록"],
    "clarification_question": "재질문 내용 (needs_clarification이 true일 때만)",
    "reasoning": "판단 근거"
}
"""


class ReAskAgent(BaseAgent):
    """정보가 부족할 경우 재질문하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="reask",
            description="답변에 필요한 정보가 부족할 경우 사용자에게 재질문하는 에이전트"
        )
    
    async def check_and_ask(self, user_query: str, context: Optional[Dict[str, Any]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 필요한 정보가 충분한지 확인하고, 부족하면 재질문
        
        Args:
            user_query: 사용자 질문
            context: 추가 컨텍스트
            conversation_history: 대화 이력
            
        Returns:
            확인 결과 및 재질문 (필요한 경우)
        """
        self.logger.info(
            f"🔍 정보 충분성 확인 시작",
            {
                "query": user_query[:100],
                "has_context": context is not None,
                "has_history": conversation_history is not None
            }
        )
        
        try:
            # 대화 이력을 포함한 프롬프트 구성
            messages = [
                {"role": "system", "content": REASK_SYSTEM_PROMPT}
            ]
            
            # 대화 이력 추가
            if conversation_history:
                for msg in conversation_history[-5:]:  # 최근 5개만 포함
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # 현재 질문 및 컨텍스트
            user_content = f"사용자 질문: {user_query}\n\n"
            if context:
                user_content += f"추가 컨텍스트:\n{self._format_context(context)}\n\n"
            user_content += "위 질문에 답변하기 위해 필요한 정보가 충분한지 확인하고, 부족한 정보가 있다면 재질문해주세요."
            
            messages.append({"role": "user", "content": user_content})
            
            # LLM 호출
            response = await self._call_llm(messages, temperature=0.3)
            
            # JSON 파싱
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 기본값 반환
                result = {
                    "needs_clarification": False,
                    "missing_info": [],
                    "clarification_question": "",
                    "reasoning": "응답 파싱 실패"
                }
            
            self.logger.info(
                f"✅ 정보 충분성 확인 완료",
                {
                    "needs_clarification": result.get("needs_clarification", False),
                    "missing_info_count": len(result.get("missing_info", []))
                }
            )
            
            return {
                "agent": self.name,
                "needs_clarification": result.get("needs_clarification", False),
                "missing_info": result.get("missing_info", []),
                "clarification_question": result.get("clarification_question", ""),
                "reasoning": result.get("reasoning", ""),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(
                f"❌ 정보 충분성 확인 실패",
                {"error": str(e)},
                exc_info=True
            )
            return {
                "agent": self.name,
                "needs_clarification": False,
                "missing_info": [],
                "clarification_question": "",
                "error": str(e),
                "status": "error"
            }
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 문자열로 포맷팅"""
        import json
        return json.dumps(context, ensure_ascii=False, indent=2)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        BaseAgent 인터페이스 구현
        """
        conversation_history = context.get("conversation_history", []) if context else []
        return await self.check_and_ask(task, context, conversation_history)


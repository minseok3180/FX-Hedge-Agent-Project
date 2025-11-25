"""ReAct 에이전트 - Reasoning과 Acting을 반복하여 문제 해결"""
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.tools.rdb_query import RDBHardTool
from src.tools.rdb_llm import RDBSoftTool


REACT_SYSTEM_PROMPT = """당신은 Reasoning과 Acting을 반복하여 문제를 해결하는 ReAct 에이전트입니다.

## ReAct 패턴
1. **Thought**: 현재 상황을 분석하고 다음 행동을 계획합니다
2. **Action**: 계획한 행동을 실행합니다 (tool 사용)
3. **Observation**: 행동의 결과를 관찰합니다
4. **Repeat**: 목표를 달성할 때까지 1-3을 반복합니다

## 사용 가능한 도구
- rdb_hard: 미리 정의된 쿼리를 사용하여 데이터 조회
- rdb_soft: LLM이 쿼리를 생성하여 데이터 조회

## 응답 형식
각 단계마다 다음 형식으로 응답하세요:
{
    "thought": "현재 상황 분석 및 다음 행동 계획",
    "action": "tool_name",
    "action_input": {"param1": "value1", "param2": "value2"},
    "observation": "도구 실행 결과",
    "final_answer": "최종 답변 (목표 달성 시)"
}

목표를 달성하면 final_answer를 포함하고, 그렇지 않으면 다음 thought와 action을 계속합니다.
"""


class ReActAgent(BaseAgent):
    """ReAct 패턴을 사용하는 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="react",
            description="Reasoning과 Acting을 반복하여 문제를 해결하는 ReAct 에이전트"
        )
        self.rdb_hard = RDBHardTool()
        self.rdb_soft = RDBSoftTool()
        self.max_iterations = 5  # 최대 반복 횟수
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ReAct 패턴으로 작업 수행
        
        Args:
            task: 수행할 작업
            context: 추가 컨텍스트
            
        Returns:
            실행 결과
        """
        self.logger.info(
            f"🔄 ReAct 에이전트 실행 시작",
            {"task": task, "max_iterations": self.max_iterations}
        )
        
        conversation_history = []
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                self.logger.debug(f"🔄 ReAct 반복 {iteration}/{self.max_iterations}")
                
                # 현재까지의 대화 이력 포함
                messages = [
                    {"role": "system", "content": REACT_SYSTEM_PROMPT}
                ]
                
                # 대화 이력 추가
                if conversation_history:
                    for msg in conversation_history:
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                
                # 현재 작업 및 컨텍스트
                user_content = f"작업: {task}\n\n"
                if context:
                    user_content += f"컨텍스트: {self._format_context(context)}\n\n"
                user_content += "위 작업을 ReAct 패턴으로 수행하세요."
                
                messages.append({"role": "user", "content": user_content})
                
                # LLM 호출
                response = await self._call_llm(messages, temperature=0.3)
                
                # JSON 파싱
                import json
                try:
                    step_result = json.loads(response)
                except json.JSONDecodeError:
                    step_result = {
                        "thought": response,
                        "action": None,
                        "action_input": {},
                        "observation": "",
                        "final_answer": ""
                    }
                
                # Thought 기록
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Thought: {step_result.get('thought', '')}"
                })
                
                # Action 실행
                action = step_result.get("action")
                if action:
                    observation = await self._execute_action(action, step_result.get("action_input", {}))
                    step_result["observation"] = observation
                    
                    # Observation 기록
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Action: {action}\nObservation: {observation}"
                    })
                else:
                    observation = "No action specified"
                
                # 최종 답변이 있으면 종료
                final_answer = step_result.get("final_answer", "")
                if final_answer:
                    self.logger.info(
                        f"✅ ReAct 에이전트 완료",
                        {"iterations": iteration, "final_answer_length": len(final_answer)}
                    )
                    return {
                        "agent": self.name,
                        "task": task,
                        "final_answer": final_answer,
                        "iterations": iteration,
                        "conversation_history": conversation_history,
                        "status": "success"
                    }
            
            # 최대 반복 횟수 도달
            self.logger.warning(
                f"⚠️ ReAct 에이전트 최대 반복 횟수 도달",
                {"iterations": iteration}
            )
            return {
                "agent": self.name,
                "task": task,
                "final_answer": "최대 반복 횟수에 도달했습니다. 작업을 완료하지 못했습니다.",
                "iterations": iteration,
                "conversation_history": conversation_history,
                "status": "max_iterations_reached"
            }
            
        except Exception as e:
            self.logger.error(
                f"❌ ReAct 에이전트 실행 실패",
                {"error": str(e), "iterations": iteration},
                exc_info=True
            )
            return {
                "agent": self.name,
                "task": task,
                "error": str(e),
                "iterations": iteration,
                "conversation_history": conversation_history,
                "status": "error"
            }
    
    async def _execute_action(self, action: str, action_input: Dict[str, Any]) -> str:
        """액션 실행"""
        try:
            if action == "rdb_hard":
                query_key = action_input.get("query_key", "get_latest")
                params = action_input.get("params", ())
                results = await self.rdb_hard.execute(query_key, params if params else None)
                return f"조회된 데이터: {len(results)}개 결과"
            
            elif action == "rdb_soft":
                user_request = action_input.get("user_request", "")
                results = await self.rdb_soft.generate_and_execute(user_request)
                return f"조회된 데이터: {results.get('results_count', 0)}개 결과"
            
            else:
                return f"알 수 없는 액션: {action}"
                
        except Exception as e:
            return f"액션 실행 실패: {str(e)}"
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 문자열로 포맷팅"""
        import json
        return json.dumps(context, ensure_ascii=False, indent=2)


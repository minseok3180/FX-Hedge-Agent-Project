from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# .env를 먼저 로드 (base_agent가 settings를 import하기 전에)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)
else:
    load_dotenv(override=False)

from .base_agent import BaseAgent
from src.config.settings_temp import settings_temp

HedgeAction = Literal["NO_HEDGE", "HEDGE_50", "HEDGE_100"]
HedgeInstrument = Literal["FORWARD", "HEDGED_ETF"]
RiskProfile = Literal["conservative", "neutral", "aggressive"]
ExposureType = Literal["FX_CASH", "FOREIGN_ASSET"]


class UserProfile(BaseModel):
    risk_profile: RiskProfile
    exposure_type: ExposureType
    exposure_amount: float = Field(gt=0)
    horizon_months: int = Field(gt=0)


class HedgeInput(BaseModel):
    user_profile: UserProfile
    risk_score: float
    regime_score: float
    vol_score: float
    base_instrument: HedgeInstrument
    news_context: Optional[str] = None
    concept_docs: List[str] = Field(default_factory=list)


class BaseHedgeResult(BaseModel):
    hedge_ratio_base: float
    base_action: HedgeAction


class FinalActionResult(BaseModel):
    final_action: HedgeAction
    override: bool
    reason: str


class HedgeResponse(BaseModel):
    hedge_ratio_base: float
    base_action: HedgeAction
    final_action: HedgeAction
    base_instrument: HedgeInstrument
    final_instrument: HedgeInstrument
    instrument_override: bool
    reasoning: Dict[str, str]


class HedgeLLMInterface(Protocol):
    async def select_final_action(
        self,
        hedge_input: HedgeInput,
        base_result: BaseHedgeResult,
    ) -> FinalActionResult:
        ...


class OpenAIHedgeLLMAdapter:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def select_final_action(
        self,
        hedge_input: HedgeInput,
        base_result: BaseHedgeResult,
    ) -> FinalActionResult:
        system_prompt = (
            "You are an FX hedge strategist. Decide whether to keep the base "
            "hedge action or override it based on the provided context. "
            "Respond in JSON with keys final_action, override, reason."
        )
        user_payload = {
            "hedge_ratio_base": base_result.hedge_ratio_base,
            "base_action": base_result.base_action,
            "concept_docs": hedge_input.concept_docs,
            "news_context": hedge_input.news_context,
            "user_profile": hedge_input.user_profile.model_dump(),
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ]
        llm_response = await self.agent._call_llm(messages, temperature=0.2)
        return self._parse_response(llm_response, base_result)

    def _parse_response(
        self,
        llm_response: str,
        fallback: BaseHedgeResult,
    ) -> FinalActionResult:
        final_action: HedgeAction = fallback.base_action
        override = False
        reason = llm_response.strip() or "LLM response empty; kept base action."

        try:
            data = json.loads(llm_response)
            action = data.get("final_action")
            if action in {"NO_HEDGE", "HEDGE_50", "HEDGE_100"}:
                final_action = action  # type: ignore[assignment]
                override = final_action != fallback.base_action
            reason = data.get("reason", reason)
            if isinstance(data.get("override"), bool):
                override = data["override"]
        except json.JSONDecodeError:
            pass

        return FinalActionResult(
            final_action=final_action,
            override=override,
            reason=reason,
        )


class RiskStrategyAgent(BaseAgent):
    def __init__(self, llm_adapter: Optional[HedgeLLMInterface] = None):
        self.settings = settings_temp
        super().__init__(
            name="RiskStrategyAgent",
            description="Computes base hedge ratios and routes decisions to LLM.",
        )
        self.llm_adapter = llm_adapter or OpenAIHedgeLLMAdapter(self)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if not context or "hedge_input" not in context:
            raise ValueError("context must include 'hedge_input'.")

        hedge_input = context["hedge_input"]
        if isinstance(hedge_input, dict):
            hedge_input = HedgeInput.model_validate(hedge_input)
        elif not isinstance(hedge_input, HedgeInput):
            raise TypeError("hedge_input must be a HedgeInput or dict.")

        response = await self.generate_strategy(hedge_input)
        return response.model_dump()

    async def generate_strategy(self, hedge_input: HedgeInput) -> HedgeResponse:
        base_result = self._compute_base_hedge(hedge_input)
        final_action = await self.llm_adapter.select_final_action(
            hedge_input=hedge_input,
            base_result=base_result,
        )

        final_instrument = hedge_input.base_instrument
        instrument_override = False
        reasoning = {
            "base_reason": (
                f"Base action {base_result.base_action} derived from "
                f"hedge ratio {base_result.hedge_ratio_base:.2f}."
            ),
            "llm_reason": final_action.reason,
            "instrument_reason": (
                "Instrument selection locked to base instrument in MVP."
            ),
        }

        return HedgeResponse(
            hedge_ratio_base=base_result.hedge_ratio_base,
            base_action=base_result.base_action,
            final_action=final_action.final_action,
            base_instrument=hedge_input.base_instrument,
            final_instrument=final_instrument,
            instrument_override=instrument_override,
            reasoning=reasoning,
        )

    def _compute_base_hedge(self, hedge_input: HedgeInput) -> BaseHedgeResult:
        ratio = (
            hedge_input.risk_score
            * hedge_input.regime_score
            * hedge_input.vol_score
        )
        hedge_ratio = self._clip_ratio(ratio)
        base_action = self._map_ratio_to_action(hedge_ratio)
        return BaseHedgeResult(
            hedge_ratio_base=hedge_ratio,
            base_action=base_action,
        )

    @staticmethod
    def _clip_ratio(ratio: float) -> float:
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _map_ratio_to_action(ratio: float) -> HedgeAction:
        if ratio < 0.3:
            return "NO_HEDGE"
        if ratio < 0.7:
            return "HEDGE_50"
        return "HEDGE_100"


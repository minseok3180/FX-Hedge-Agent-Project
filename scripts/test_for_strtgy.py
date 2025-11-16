import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# .env를 가장 먼저 로드 (다른 모듈들이 settings를 import하기 전에)
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)
else:
    load_dotenv(override=False)

# base_agent를 먼저 직접 로드 (__init__.py를 거치지 않음)
import importlib.util

# base_agent 로드
base_agent_path = PROJECT_ROOT / "src" / "agents" / "base_agent.py"
base_agent_spec = importlib.util.spec_from_file_location("src.agents.base_agent", base_agent_path)
base_agent_module = importlib.util.module_from_spec(base_agent_spec)
if "src.agents.base_agent" not in sys.modules:
    sys.modules["src.agents.base_agent"] = base_agent_module
base_agent_spec.loader.exec_module(base_agent_module)

# risk_strategy_agent 로드
risk_strategy_path = PROJECT_ROOT / "src" / "agents" / "risk_strategy_agent.py"
risk_strategy_spec = importlib.util.spec_from_file_location("src.agents.risk_strategy_agent", risk_strategy_path)
risk_strategy_module = importlib.util.module_from_spec(risk_strategy_spec)
if "src.agents.risk_strategy_agent" not in sys.modules:
    sys.modules["src.agents.risk_strategy_agent"] = risk_strategy_module
risk_strategy_spec.loader.exec_module(risk_strategy_module)

RiskStrategyAgent = risk_strategy_module.RiskStrategyAgent
HedgeInput = risk_strategy_module.HedgeInput
UserProfile = risk_strategy_module.UserProfile

async def main():
    agent = RiskStrategyAgent()
    hedge_input = HedgeInput(
        user_profile=UserProfile(
            risk_profile="neutral",
            exposure_type="FX_CASH",
            exposure_amount=1_000_000,
            horizon_months=12,
        ),
        risk_score=0.4,
        regime_score=0.8,
        vol_score=1.1,
        base_instrument="FORWARD",
        news_context="USD strength cooling, EUR stable.",
        concept_docs=["Hedge basics...", "FX cash flow hedging..."],
    )
    response = await agent.generate_strategy(hedge_input)
    print(response.model_dump())

if __name__ == "__main__":
    asyncio.run(main())
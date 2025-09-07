#!/bin/bash
# CLI 모드 실행 스크립트

# 가상환경 활성화
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source fx_hedge_env/Scripts/activate
else
    source fx_hedge_env/bin/activate
fi

echo "💰 외환 헷지전략 에이전트 CLI 모드"
echo "----------------------------------------"

python src/main.py --mode interactive

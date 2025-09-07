#!/bin/bash
# Streamlit 앱 실행 스크립트

# 가상환경 활성화
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source fx_hedge_env/Scripts/activate
else
    source fx_hedge_env/bin/activate
fi

echo "🚀 외환 헷지전략 에이전트 Streamlit 앱 시작..."
echo "📱 브라우저에서 http://localhost:8501 을 열어주세요"
echo "⏹️  종료하려면 Ctrl+C를 누르세요"
echo "----------------------------------------"

# 환경변수 설정 (torch.classes 오류 해결)
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

streamlit run src/streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

#!/usr/bin/env python3
"""
외환 헷지전략 에이전트 Streamlit 앱 실행 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """필요한 패키지 설치 확인"""
    try:
        import streamlit
        import plotly
        print("✅ 필요한 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 없습니다: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install streamlit plotly")
        return False

def run_streamlit_app():
    """Streamlit 앱 실행"""
    # 현재 디렉토리를 프로젝트 루트로 설정
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # 환경변수 설정 (torch.classes 오류 해결)
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Streamlit 앱 실행
    try:
        print("🚀 외환 헷지전략 에이전트 Streamlit 앱을 시작합니다...")
        print("📱 브라우저에서 http://localhost:8501 을 열어주세요")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 앱이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 앱 실행 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("=" * 60)
    print("💰 외환 헷지전략 에이전트 Streamlit 앱")
    print("🤖 KT Mi:dm 2.0 모델 기반")
    print("=" * 60)
    
    # 패키지 확인
    if not check_requirements():
        sys.exit(1)
    
    # 앱 실행
    run_streamlit_app()

if __name__ == "__main__":
    main()

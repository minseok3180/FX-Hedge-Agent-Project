@echo off
REM 외환 헷지전략 에이전트 가상환경 설정 스크립트 (Windows)
REM KT Mi:dm 2.0 모델 기반 멀티 에이전트 시스템

setlocal enabledelayedexpansion

echo ==========================================
echo 💰 외환 헷지전략 에이전트 환경 설정
echo 🤖 KT Mi:dm 2.0 모델 기반
echo ==========================================

REM Python 버전 확인
echo [INFO] Python 버전 확인 중...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python이 설치되지 않았습니다. Python 3.11+ 설치 후 다시 시도하세요.
    pause
    exit /b 1
)

python --version
echo [SUCCESS] Python 발견

REM 가상환경 생성
echo [INFO] 가상환경 생성 중...
set VENV_NAME=fx_hedge_env

if exist "%VENV_NAME%" (
    echo [WARNING] 가상환경 '%VENV_NAME%'이 이미 존재합니다.
    set /p choice="기존 가상환경을 삭제하고 새로 생성하시겠습니까? (y/N): "
    if /i "%choice%"=="y" (
        echo [INFO] 기존 가상환경 삭제 중...
        rmdir /s /q "%VENV_NAME%"
    ) else (
        echo [INFO] 기존 가상환경 사용
        goto :activate_venv
    )
)

python -m venv "%VENV_NAME%"
if %errorlevel% neq 0 (
    echo [ERROR] 가상환경 생성 실패
    pause
    exit /b 1
)
echo [SUCCESS] 가상환경 '%VENV_NAME%' 생성 완료

:activate_venv
REM 가상환경 활성화
echo [INFO] 가상환경 활성화 중...
call "%VENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] 가상환경 활성화 실패
    pause
    exit /b 1
)
echo [SUCCESS] 가상환경 활성화 완료

REM pip 업그레이드
echo [INFO] pip 업그레이드 중...
python -m pip install --upgrade pip
echo [SUCCESS] pip 업그레이드 완료

REM 기본 패키지 설치
echo [INFO] 기본 패키지 설치 중...
pip install wheel setuptools
pip install streamlit plotly
pip install aiohttp httpx
pip install pandas numpy
pip install python-dotenv
pip install structlog
pip install transformers torch sentence-transformers
pip install chromadb
pip install tqdm requests beautifulsoup4
echo [SUCCESS] 기본 패키지 설치 완료

REM requirements.txt 설치
echo [INFO] requirements.txt 패키지 설치 중...
if exist "requirements.txt" (
    pip install -r requirements.txt
    echo [SUCCESS] requirements.txt 설치 완료
) else (
    echo [WARNING] requirements.txt 파일을 찾을 수 없습니다.
)

REM 환경변수 파일 생성
echo [INFO] 환경변수 파일 생성 중...
if exist ".env" (
    echo [WARNING] .env 파일이 이미 존재합니다.
    set /p choice="기존 파일을 덮어쓰시겠습니까? (y/N): "
    if /i not "%choice%"=="y" (
        echo [INFO] 기존 .env 파일 유지
        goto :create_scripts
    )
)

(
echo # 외환 헷지전략 에이전트 환경변수 설정 ^(로컬 환경용^)
echo.
echo # LLM 설정 ^(로컬 Mi:dm 2.0^)
echo LLM_MODEL_NAME=K-intelligence/Midm-2.0-Base-Instruct
echo LLM_USE_LOCAL=true
echo LLM_DEVICE=auto
echo LLM_MAX_TOKENS=4096
echo LLM_TEMPERATURE=0.7
echo LLM_TOP_P=0.9
echo LLM_FREQUENCY_PENALTY=0.0
echo LLM_PRESENCE_PENALTY=0.0
echo.
echo # 임베딩 모델 ^(로컬^)
echo EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
echo EMBEDDING_DIMENSION=768
echo.
echo # 데이터베이스 설정 ^(로컬 환경^)
echo # TSDB ^(Time Series Database^) - 추후 개발
echo TSDB_HOST=localhost
echo TSDB_PORT=8086
echo TSDB_DATABASE=fx_timeseries
echo TSDB_USERNAME=admin
echo TSDB_PASSWORD=password
echo.
echo # VDB ^(Vector Database^) - 로컬 ChromaDB
echo VDB_TYPE=chromadb
echo VDB_PATH=./data/chromadb
echo VDB_COLLECTION=hedge_strategies
echo.
echo # RDB ^(Relational Database^) - CSV 파일로 대체
echo RDB_TYPE=csv
echo CSV_DATA_DIR=./data/csv
echo USER_DATA_FILE=users.csv
echo TRADING_HISTORY_FILE=trading_history.csv
echo PORTFOLIO_FILE=portfolios.csv
echo.
echo # 에이전트 설정
echo SEARCH_TIMEOUT=30
echo MAX_NEWS_COUNT=100
echo TSDB_QUERY_LIMIT=1000
echo MAX_STRATEGY_COUNT=50
echo TRADING_TIMEOUT=60
echo MAX_ITERATIONS=10
echo CONVERSATION_TIMEOUT=300
) > .env

echo [SUCCESS] .env 파일 생성 완료
echo [WARNING] ⚠️  로컬 환경에서는 API 키가 필요하지 않습니다!

:create_scripts
REM 실행 스크립트 생성
echo [INFO] 실행 스크립트 생성 중...

REM Streamlit 앱 실행 스크립트
(
echo @echo off
echo REM Streamlit 앱 실행 스크립트
echo.
echo REM 가상환경 활성화
echo call fx_hedge_env\Scripts\activate.bat
echo.
echo echo 🚀 외환 헷지전략 에이전트 Streamlit 앱 시작...
echo echo 📱 브라우저에서 http://localhost:8501 을 열어주세요
echo echo ⏹️  종료하려면 Ctrl+C를 누르세요
echo echo ----------------------------------------
echo.
echo streamlit run src\streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
echo pause
) > run_streamlit.bat

REM CLI 실행 스크립트
(
echo @echo off
echo REM CLI 모드 실행 스크립트
echo.
echo REM 가상환경 활성화
echo call fx_hedge_env\Scripts\activate.bat
echo.
echo echo 💰 외환 헷지전략 에이전트 CLI 모드
echo echo ----------------------------------------
echo.
echo python src\main.py --mode interactive
echo pause
) > run_cli.bat

echo [SUCCESS] 실행 스크립트 생성 완료

REM 설정 검증
echo [INFO] 설정 검증 중...
call "%VENV_NAME%\Scripts\activate.bat"
python -c "import streamlit; print('[SUCCESS] Streamlit 설치 확인')" 2>nul || echo [ERROR] Streamlit 설치 실패
python -c "import pandas; print('[SUCCESS] Pandas 설치 확인')" 2>nul || echo [ERROR] Pandas 설치 실패
python -c "import aiohttp; print('[SUCCESS] aiohttp 설치 확인')" 2>nul || echo [ERROR] aiohttp 설치 실패

echo [SUCCESS] 설정 검증 완료

REM 사용법 안내
echo.
echo ==========================================
echo 🎉 설치 완료!
echo ==========================================
echo.
echo 📋 사용 방법:
echo.
echo 1. 환경변수 설정:
echo    notepad .env  # API 키를 실제 값으로 수정
echo.
echo 2. Streamlit 웹 앱 실행:
echo    run_streamlit.bat
echo    또는
echo    python run_app.py
echo.
echo 3. CLI 모드 실행:
echo    run_cli.bat
echo    또는
echo    python src\main.py --mode interactive
echo.
echo 4. 가상환경 비활성화:
echo    deactivate
echo.
echo 🌐 웹 앱 접속: http://localhost:8501
echo.
echo ⚠️  주의사항:
echo    - .env 파일에서 API 키를 실제 값으로 수정해주세요
echo    - 데이터베이스 서버가 실행 중인지 확인해주세요
echo.
pause

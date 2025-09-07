#!/bin/bash

# 외환 헷지전략 에이전트 가상환경 설정 스크립트
# KT Mi:dm 2.0 모델 기반 멀티 에이전트 시스템

set -e  # 오류 발생 시 스크립트 중단

echo "=========================================="
echo "💰 외환 헷지전략 에이전트 환경 설정"
echo "🤖 KT Mi:dm 2.0 모델 기반"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Python 버전 확인
check_python() {
    print_status "Python 버전 확인 중..."
    
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        print_success "Python 3.11 발견"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$PYTHON_VERSION" == "3.1"* ]] || [[ "$PYTHON_VERSION" == "3.1"* ]]; then
            PYTHON_CMD="python3"
            print_success "Python $PYTHON_VERSION 발견"
        else
            print_warning "Python 3.11+ 권장 (현재: $PYTHON_VERSION)"
            PYTHON_CMD="python3"
        fi
    else
        print_error "Python이 설치되지 않았습니다. Python 3.11+ 설치 후 다시 시도하세요."
        exit 1
    fi
}

# 가상환경 생성
create_venv() {
    print_status "가상환경 생성 중..."
    
    VENV_NAME="fx_hedge_env"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "가상환경 '$VENV_NAME'이 이미 존재합니다."
        read -p "기존 가상환경을 삭제하고 새로 생성하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "기존 가상환경 삭제 중..."
            rm -rf "$VENV_NAME"
        else
            print_status "기존 가상환경 사용"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    print_success "가상환경 '$VENV_NAME' 생성 완료"
}

# 가상환경 활성화
activate_venv() {
    print_status "가상환경 활성화 중..."
    
    VENV_NAME="fx_hedge_env"
    
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        source "$VENV_NAME/Scripts/activate"
    else
        # macOS/Linux
        source "$VENV_NAME/bin/activate"
    fi
    
    print_success "가상환경 활성화 완료"
}

# pip 업그레이드
upgrade_pip() {
    print_status "pip 업그레이드 중..."
    pip install --upgrade pip
    print_success "pip 업그레이드 완료"
}

# 필수 패키지 설치
install_packages() {
    print_status "필수 패키지 설치 중..."
    
    # 기본 패키지들
    pip install wheel setuptools
    
    # Streamlit 및 웹 관련
    pip install streamlit plotly
    
    # 비동기 처리
    pip install aiohttp httpx
    
    # 데이터 처리
    pip install pandas numpy
    
    # 환경 설정
    pip install python-dotenv
    
    # 로깅
    pip install structlog
    
    # 로컬 LLM 및 임베딩 모델
    pip install transformers torch sentence-transformers
    
    # 로컬 ChromaDB
    pip install chromadb
    
    # 추가 유틸리티
    pip install tqdm requests beautifulsoup4
    
    print_success "기본 패키지 설치 완료"
}

# requirements.txt 설치
install_requirements() {
    print_status "requirements.txt 패키지 설치 중..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "requirements.txt 설치 완료"
    else
        print_warning "requirements.txt 파일을 찾을 수 없습니다."
    fi
}

# 환경변수 파일 생성
create_env_file() {
    print_status "환경변수 파일 생성 중..."
    
    ENV_FILE=".env"
    
    if [ -f "$ENV_FILE" ]; then
        print_warning ".env 파일이 이미 존재합니다."
        read -p "기존 파일을 덮어쓰시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "기존 .env 파일 유지"
            return 0
        fi
    fi
    
    cat > "$ENV_FILE" << EOF
# 외환 헷지전략 에이전트 환경변수 설정 (로컬 환경용)

# LLM 설정 (로컬 Mi:dm 2.0)
LLM_MODEL_NAME=K-intelligence/Midm-2.0-Mini-Instruct
LLM_USE_LOCAL=true
LLM_DEVICE=auto
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_FREQUENCY_PENALTY=0.0
LLM_PRESENCE_PENALTY=0.0

# 임베딩 모델 (로컬)
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
EMBEDDING_DIMENSION=768

# 데이터베이스 설정 (로컬 환경)
# TSDB (Time Series Database) - 추후 개발
TSDB_HOST=localhost
TSDB_PORT=8086
TSDB_DATABASE=fx_timeseries
TSDB_USERNAME=admin
TSDB_PASSWORD=password

# VDB (Vector Database) - 로컬 ChromaDB
VDB_TYPE=chromadb
VDB_PATH=./data/chromadb
VDB_COLLECTION=hedge_strategies

# RDB (Relational Database) - CSV 파일로 대체
RDB_TYPE=csv
CSV_DATA_DIR=./data/csv
USER_DATA_FILE=users.csv
TRADING_HISTORY_FILE=trading_history.csv
PORTFOLIO_FILE=portfolios.csv

# 에이전트 설정
SEARCH_TIMEOUT=30
MAX_NEWS_COUNT=100
TSDB_QUERY_LIMIT=1000
MAX_STRATEGY_COUNT=50
TRADING_TIMEOUT=60
MAX_ITERATIONS=10
CONVERSATION_TIMEOUT=300

# 성능 최적화 설정
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
HF_HUB_DISABLE_SYMLINKS_WARNING=1
EOF
    
    print_success ".env 파일 생성 완료"
    print_warning "⚠️  로컬 환경에서는 API 키가 필요하지 않습니다!"
}

# 실행 스크립트 생성
create_run_scripts() {
    print_status "실행 스크립트 생성 중..."
    
    # Streamlit 앱 실행 스크립트
    cat > run_streamlit.sh << 'EOF'
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

streamlit run src/streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
EOF
    
    # CLI 실행 스크립트
    cat > run_cli.sh << 'EOF'
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
EOF
    
    # 실행 권한 부여
    chmod +x run_streamlit.sh run_cli.sh
    
    print_success "실행 스크립트 생성 완료"
}

# 설정 검증
verify_setup() {
    print_status "설정 검증 중..."
    
    # 가상환경 활성화
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source fx_hedge_env/Scripts/activate
    else
        source fx_hedge_env/bin/activate
    fi
    
    # Python 패키지 확인
    python -c "import streamlit; print('✅ Streamlit 설치 확인')" 2>/dev/null || print_error "Streamlit 설치 실패"
    python -c "import pandas; print('✅ Pandas 설치 확인')" 2>/dev/null || print_error "Pandas 설치 실패"
    python -c "import aiohttp; print('✅ aiohttp 설치 확인')" 2>/dev/null || print_error "aiohttp 설치 실패"
    
    print_success "설정 검증 완료"
}

# 사용법 안내
show_usage() {
    echo ""
    echo "=========================================="
    echo "🎉 설치 완료!"
    echo "=========================================="
    echo ""
    echo "📋 사용 방법:"
    echo ""
    echo "1. 환경변수 설정:"
    echo "   nano .env  # API 키를 실제 값으로 수정"
    echo ""
    echo "2. Streamlit 웹 앱 실행:"
    echo "   ./run_streamlit.sh"
    echo "   또는"
    echo "   python run_app.py"
    echo ""
    echo "3. CLI 모드 실행:"
    echo "   ./run_cli.sh"
    echo "   또는"
    echo "   python src/main.py --mode interactive"
    echo ""
    echo "4. 가상환경 비활성화:"
    echo "   deactivate"
    echo ""
    echo "🌐 웹 앱 접속: http://localhost:8501"
    echo ""
    echo "⚠️  주의사항:"
    echo "   - .env 파일에서 API 키를 실제 값으로 수정해주세요"
    echo "   - 데이터베이스 서버가 실행 중인지 확인해주세요"
    echo ""
}

# 메인 실행
main() {
    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_packages
    install_requirements
    create_env_file
    create_run_scripts
    verify_setup
    show_usage
}

# 스크립트 실행
main "$@"

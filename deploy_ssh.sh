#!/bin/bash

# SSH 서버 배포 스크립트
# 사용법: ./deploy_ssh.sh username@server_ip

if [ $# -eq 0 ]; then
    echo "사용법: $0 username@server_ip"
    echo "예시: $0 user@192.168.1.100"
    exit 1
fi

SERVER="$1"
PROJECT_NAME="FX-Hedge-Agent-Project"
REMOTE_PATH="~/"

echo "=========================================="
echo "🚀 SSH 서버 배포 시작"
echo "서버: $SERVER"
echo "=========================================="

# 로컬 프로젝트 압축
echo "[INFO] 프로젝트 압축 중..."
tar -czf ${PROJECT_NAME}.tar.gz --exclude='fx_hedge_env' --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' .

# SSH 서버로 전송
echo "[INFO] 서버로 전송 중..."
scp ${PROJECT_NAME}.tar.gz $SERVER:${REMOTE_PATH}

# SSH 서버에서 설정 실행
echo "[INFO] 서버에서 설정 실행 중..."
ssh $SERVER << 'EOF'
    # 프로젝트 압축 해제
    tar -xzf FX-Hedge-Agent-Project.tar.gz
    cd FX-Hedge-Agent-Project
    
    # 실행 권한 부여
    chmod +x setup_env.sh
    
    # 환경 설정 실행
    ./setup_env.sh
    
    echo "[SUCCESS] 서버 설정 완료!"
    echo "실행 명령어:"
    echo "  cd FX-Hedge-Agent-Project"
    echo "  source fx_hedge_env/bin/activate"
    echo "  streamlit run src/streamlit_app.py --server.port 8501 --server.headless true"
EOF

# 로컬 압축 파일 삭제
rm ${PROJECT_NAME}.tar.gz

echo "=========================================="
echo "🎉 배포 완료!"
echo "=========================================="
echo ""
echo "SSH 터널 생성:"
echo "  ssh -L 8501:localhost:8501 $SERVER"
echo ""
echo "브라우저 접속:"
echo "  http://localhost:8501"
echo ""


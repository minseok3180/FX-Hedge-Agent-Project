FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY api/ ./api/
COPY config.json ./

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 빌드 시 환경 변수 주입 (ARG로 받아서 ENV로 설정)
ARG OPENAI_API_KEY
ARG OPENAI_MODEL
ARG DB_HOST
ARG DB_PORT
ARG DB_USER
ARG DB_PASSWORD
ARG DB_NAME
ARG QDRANT_HOST
ARG QDRANT_PORT
ARG QDRANT_API_KEY
ARG WEB_SEARCH_API_KEY
ARG WEB_SEARCH_ENGINE_ID
ARG API_HOST
ARG API_PORT

ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_MODEL=${OPENAI_MODEL}
ENV DB_HOST=${DB_HOST}
ENV DB_PORT=${DB_PORT}
ENV DB_USER=${DB_USER}
ENV DB_PASSWORD=${DB_PASSWORD}
ENV DB_NAME=${DB_NAME}
ENV QDRANT_HOST=${QDRANT_HOST}
ENV QDRANT_PORT=${QDRANT_PORT}
ENV QDRANT_API_KEY=${QDRANT_API_KEY}
ENV WEB_SEARCH_API_KEY=${WEB_SEARCH_API_KEY}
ENV WEB_SEARCH_ENGINE_ID=${WEB_SEARCH_ENGINE_ID}
ENV API_HOST=${API_HOST}
ENV API_PORT=${API_PORT}

# FastAPI 실행
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


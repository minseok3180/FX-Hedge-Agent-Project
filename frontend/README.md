# Frontend

프론트엔드 애플리케이션 디렉토리

## 구조

```
frontend/
├── public/          # 정적 파일 (favicon, images 등)
├── src/             # 소스 코드
│   ├── components/  # 재사용 가능한 컴포넌트
│   ├── pages/       # 페이지 컴포넌트
│   ├── hooks/       # 커스텀 React 훅
│   ├── services/    # API 서비스 (백엔드 연동)
│   ├── utils/       # 유틸리티 함수
│   └── styles/      # 전역 스타일, 테마
├── package.json     # 의존성 관리
└── README.md
```

## 기술 스택 (예시)

다음 중 선택 가능:
- **React** + TypeScript + Vite
- **Next.js** (SSR 지원)
- **Vue.js** + TypeScript
- 기타 선호하는 프레임워크

## API 연동

백엔드 API 엔드포인트: `http://localhost:8000`

주요 엔드포인트:
- `POST /chat`: 채팅 메시지 전송
- `GET /agents`: 사용 가능한 에이전트 목록
- `POST /agent/{agent_name}`: 특정 에이전트에 직접 요청

## 개발 시작

```bash
cd frontend
npm install  # 또는 yarn install
npm run dev  # 개발 서버 실행
```

## 환경 변수

프론트엔드에서 사용할 환경 변수는 `.env.local` 파일에 정의:

```env
VITE_API_URL=http://localhost:8000
# 또는
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 참고

- API 서비스는 `src/services/` 디렉토리에 구현
- 컴포넌트는 재사용 가능하도록 `src/components/`에 분리
- 페이지는 `src/pages/` 또는 라우팅 구조에 따라 구성

